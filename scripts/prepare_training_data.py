"""Freeze train/dev/calibration/fresh test before any adapter fitting."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import urllib.request
from prepare_benchmark import REPO, REVISION, SOURCES, decode_row, read_jsonl, write_jsonl, digest
from shingi.decision import LETTERS, describe, prompt_for
from shingi.metrics import human_distribution

SEED='shingi-training-v1-20260922'
# Use sources with explicit open licenses in the publisher's manifest.
# This records provenance, not a legal conclusion about a future release.
TRAIN_SOURCES=('banking77','clinc150','mmlu','arc_challenge','helpsteer2_helpfulness',
               'measuring_hate_speech','boolq','civil_comments')


def select(rows, count, excluded_ids, excluded_states, tag):
    chosen=[]
    for row in sorted(rows,key=lambda r:digest(SEED+'|'+tag+'|'+r['id'])):
        if row['id'] in excluded_ids or row['state_sha256'] in excluded_states: continue
        chosen.append(row);excluded_ids.add(row['id']);excluded_states.add(row['state_sha256'])
        if len(chosen)==count:break
    if len(chosen)!=count:raise ValueError(f'{tag}: need {count}, found {len(chosen)}')
    return chosen


def training_prompt(row, variant):
    q=row['question'];kind=q['type'];instructions=q['instructions'];gold=str(row['label'])
    human=human_distribution(row)
    if kind=='choice':options=list(q['criteria'].items())
    elif kind=='score':
        options=[(str(i),v) for i,v in enumerate(q['criteria'])]
        # Numeric keys preserve ordinal meaning even when display order changes.
        instructions=describe(instructions)+' Select the level whose description best fits. Level numbers preserve their original order.'
    else:
        criteria=q.get('criteria') or {}
        options=[('yes',criteria.get('true') or 'The statement is true.'),('no',criteria.get('false') or 'The statement is false.')]
        gold='yes' if gold=='1' else 'no'
        if human is not None:human={'yes':human['1'],'no':human['0']}
    rng=random.Random(SEED+'|'+row['id']+'|'+str(variant))
    if len(options)>52:
        size=rng.choice([4,8,16,32])
        positive=[x for x in options if x[0]==gold]
        if len(positive)!=1:raise ValueError('missing gold choice')
        options=positive+rng.sample([x for x in options if x[0]!=gold],size-1)
    rng.shuffle(options)
    keys=[x[0] for x in options]
    if human is None:target=[float(key==gold) for key in keys]
    else:
        target=[human.get(key,0.) for key in keys];total=sum(target)
        if total<=0:raise ValueError('empty target probability')
        target=[p/total for p in target]
    return {'id':row['id']+f'/p{variant}','record_id':row['id'],'source':row['source'],
            'primitive':kind,'state_sha256':row['state_sha256'], 'option_keys':keys,
            'prompt':prompt_for(row['state'],instructions,options),'labels':list(LETTERS[:len(options)]),
            'target':target,'hard_target':keys.index(gold)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,default=Path('artifacts/benchmark-v1'))
    p.add_argument('--output',type=Path,default=Path('artifacts/decision-v2'));a=p.parse_args()
    if a.output.exists():raise SystemExit('preserve existing dataset; choose new directory')
    a.output.mkdir(parents=True);raw=a.output/'raw';raw.mkdir()
    old=read_jsonl(a.baseline/'test.jsonl')+read_jsonl(a.baseline/'calibration.jsonl')
    ids={r['id'] for r in old};states={r['state_sha256'] for r in old}
    pools={}
    for source in SOURCES:
        for split in ('test','validation'):
            path=a.baseline/'raw'/'data'/source/(split+'.jsonl')
            if path.exists():pools[source,split]=[decode_row(r) for r in read_jsonl(path)]
    test=[]
    for source in SOURCES:test+=select(pools[source,'test'],100,ids,states,source+'/fresh-test')
    dev=[];calibration=[]
    for source in TRAIN_SOURCES:
        dev+=select(pools[source,'validation'],32,ids,states,source+'/dev')
        calibration+=select(pools[source,'validation'],30,ids,states,source+'/calibration')
    # Protect ALL published test/validation states, including unselected records.
    forbidden_ids={r['id'] for pool in pools.values() for r in pool}|{r['id'] for r in old}
    forbidden_states={r['state_sha256'] for pool in pools.values() for r in pool}|{r['state_sha256'] for r in old}
    train_ids=set(forbidden_ids);train_states=set(forbidden_states);train=[];files={}
    for source in TRAIN_SOURCES:
        url=f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/data/{source}/train.jsonl'
        data=urllib.request.urlopen(url,timeout=120).read();path=raw/(source+'-train.jsonl');path.write_bytes(data)
        files[source]={'url':url,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        rows=[decode_row(r) for r in read_jsonl(path)]
        available=[r for r in rows if r['id'] not in train_ids and r['state_sha256'] not in train_states]
        train+=select(available,min(400,len({r['state_sha256'] for r in available})),train_ids,train_states,source+'/train')
    assert not {r['state_sha256'] for r in train}&forbidden_states
    prompts=[training_prompt(r,v) for r in train for v in (0,1)]
    prompts.sort(key=lambda r:digest(SEED+'|order|'+r['id']))
    splits={'train':train,'train-prompts':prompts,'dev':dev,'calibration':calibration,'test':test}
    manifest={'dataset':REPO,'revision':REVISION,'seed':SEED,'train_sources':list(TRAIN_SOURCES),
              'held_out_sources':sorted(set(SOURCES)-set(TRAIN_SOURCES)), 'files':files,
              'counts':{k:len(v) for k,v in splits.items()},'train_by_source':dict(Counter(r['source'] for r in train)),
              'overlap':{'id':0,'state_sha256':0},'excluded_published_eval_states':len(forbidden_states),
              'prior_baseline_manifest_sha256':hashlib.sha256((a.baseline/'manifest.json').read_bytes()).hexdigest(),
              'shuffle_ids':[r['id'] for r in sorted(test,key=lambda r:digest(SEED+'|shuffle|'+r['id'])) if r['primitive']=='choice'][:200],
              'license_scope':'Source licenses are recorded upstream; release permission review remains separate.',
              'pretraining_contamination':'Unknown; these exclusions apply to this adaptation only.'}
    for split,rows in splits.items():
        path=a.output/(split+'.jsonl');write_jsonl(path,rows);manifest[split+'_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ('counts','train_by_source','held_out_sources','overlap')},indent=2))

if __name__=='__main__':main()
