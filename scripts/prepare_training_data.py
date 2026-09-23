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
APACHE_SOURCES = tuple(s for s in TRAIN_SOURCES if s not in ('arc_challenge', 'boolq'))
APACHE_SEED = 'shingi-apache-v2-20260923'


def select(rows, count, excluded_ids, excluded_states, tag, seed=SEED):
    chosen=[]
    if count == 0: return chosen
    for row in sorted(rows,key=lambda r:digest(seed+'|'+tag+'|'+r['id'])):
        if row['id'] in excluded_ids or row['state_sha256'] in excluded_states: continue
        chosen.append(row);excluded_ids.add(row['id']);excluded_states.add(row['state_sha256'])
        if len(chosen)==count:break
    if len(chosen)!=count:raise ValueError(f'{tag}: need {count}, found {len(chosen)}')
    return chosen


def training_prompt(row, variant, seed=SEED):
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
    rng=random.Random(seed+'|'+row['id']+'|'+str(variant))
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


def prepare_apache(baseline, previous_training, previous_release, output):
    """Allowlisted fitting only; no locked test is prepared or opened here."""
    from shingi.release import sha256
    if output.exists(): raise ValueError('preserve existing dataset; choose new directory')
    old, exclusions = [], {}
    prior_training = []
    for directory, splits in ((baseline, ('test', 'calibration')),
                             (previous_training, ('train', 'dev', 'calibration', 'test')),
                             (previous_release, ('test',))):
        manifest = json.loads((directory/'manifest.json').read_text())
        if manifest['revision'] != REVISION: raise ValueError('prior revision differs')
        for split in splits:
            path = directory/(split+'.jsonl')
            if sha256(path) != manifest[split+'_sha256']: raise ValueError('prior split changed')
            records = read_jsonl(path)
            old.extend(records)
            if directory == previous_training and split == 'train': prior_training = records
            exclusions[str(path)] = sha256(path)
    ids = {r['id'] for r in old}; states = {r['state_sha256'] for r in old}
    pools, files = {}, {}
    baseline_manifest = json.loads((baseline/'manifest.json').read_text())
    for source in SOURCES:
        for split in ('test', 'validation'):
            relative = f'data/{source}/{split}.jsonl'
            path = baseline/'raw'/relative
            if not path.exists(): continue
            if sha256(path) != baseline_manifest['files'][relative]['sha256']:
                raise ValueError('pinned public data changed')
            pools[source, split] = [decode_row(r) for r in read_jsonl(path)]
    dev, calibration = [], []
    for source in APACHE_SOURCES:
        dev += select(pools[source, 'validation'], 32, ids, states, source+'/dev', APACHE_SEED)
        calibration += select(pools[source, 'validation'], 30, ids, states, source+'/calibration', APACHE_SEED)
    # Reusing allowlisted old training records is permitted. Every evaluation
    # state remains forbidden, including unselected public validation/test rows.
    prior_train_ids = {r['id'] for r in prior_training}
    forbidden = [r for r in old if r['id'] not in prior_train_ids]
    forbidden += [r for pool in pools.values() for r in pool]
    train_ids = {r['id'] for r in forbidden}; train_states = {r['state_sha256'] for r in forbidden}
    forbidden_ids, forbidden_states = train_ids.copy(), train_states.copy()
    output.mkdir(parents=True); raw = output/'raw'; raw.mkdir()
    train = []
    for source in APACHE_SOURCES:
        url = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/data/{source}/train.jsonl'
        data = urllib.request.urlopen(url, timeout=120).read()
        path = raw/(source+'-train.jsonl'); path.write_bytes(data)
        files[source] = {'url': url, 'bytes': len(data), 'sha256': sha256(path)}
        pool = [decode_row(r) for r in read_jsonl(path)]
        available = [r for r in pool if r['id'] not in train_ids and r['state_sha256'] not in train_states]
        train += select(available, min(400, len({r['state_sha256'] for r in available})),
                        train_ids, train_states, source+'/train', APACHE_SEED)
    splits = {'train': train, 'dev': dev, 'calibration': calibration}
    for name, records in splits.items():
        if set(r['source'] for r in records) - set(APACHE_SOURCES): raise ValueError('non-allowlisted fitting source')
        for other, others in splits.items():
            if name == other: continue
            if {r['id'] for r in records} & {r['id'] for r in others}: raise ValueError('split ID overlap')
            if {r['state_sha256'] for r in records} & {r['state_sha256'] for r in others}: raise ValueError('split state overlap')
    if {r['id'] for r in train} & forbidden_ids or {r['state_sha256'] for r in train} & forbidden_states:
        raise ValueError('training/evaluation overlap')
    prompts = [training_prompt(r, v, APACHE_SEED) for r in train for v in (0, 1)]
    prompts.sort(key=lambda r: digest(APACHE_SEED+'|order|'+r['id']))
    splits['train-prompts'] = prompts
    manifest = {'profile': 'apache-v2', 'dataset': REPO, 'revision': REVISION, 'seed': APACHE_SEED,
                'train_sources': list(APACHE_SOURCES), 'held_out_sources': sorted(set(SOURCES)-set(APACHE_SOURCES)),
                'files': files, 'excluded_splits': exclusions,
                'counts': {k:len(v) for k,v in splits.items()}, 'train_by_source': dict(Counter(r['source'] for r in train)),
                'overlap': {'id':0, 'state_sha256':0}, 'excluded_published_eval_states': len(forbidden_states),
                'reused_permitted_training_records': len(prior_train_ids & {r['id'] for r in train}),
                'initialization': 'Fresh LoRA on unchanged base; no prior adapter or calibration.',
                'license_scope': 'Only CC BY, MIT and CC0 sources allowed for all fitting; preserve source attribution.',
                'pretraining_contamination': 'Unknown; these exclusions apply to this adaptation only.'}
    for split, records in splits.items():
        path = output/(split+'.jsonl'); write_jsonl(path, records); manifest[split+'_sha256'] = sha256(path)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ('counts', 'train_by_source', 'overlap', 'reused_permitted_training_records')}, indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,default=Path('artifacts/benchmark-v1'))
    p.add_argument('--output',type=Path,default=Path('artifacts/decision-v2'))
    p.add_argument('--profile', choices=('original-v1', 'apache-v2'), default='original-v1')
    p.add_argument('--previous-training', type=Path); p.add_argument('--previous-release', type=Path)
    a=p.parse_args()
    if a.profile == 'apache-v2':
        if not a.previous_training or not a.previous_release: p.error('apache-v2 requires both prior dataset directories')
        return prepare_apache(a.baseline, a.previous_training, a.previous_release, a.output)
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
