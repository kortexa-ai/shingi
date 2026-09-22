"""Evaluate the frozen dev-selected adapter and unchanged base on fresh data."""
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

from shingi.backend import NativeReadout,gpu_free_mib
from shingi.calibration import fit,replay
from shingi.decision import Calibration,DecisionEngine
from shingi.metrics import report,wilson,probabilities
from train_decision_adapter import sha,rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--training',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path('artifacts/decision-v2'));p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();selection=json.loads((a.training/'selection.json').read_text());selected=selection['selected']
    if sha(selected['native'])!=selected['native_sha256']:raise RuntimeError('selected adapter changed')
    manifest=json.loads((a.data/'manifest.json').read_text())
    splits={}
    for split in ('dev','calibration','test'):
        path=a.data/(split+'.jsonl')
        if sha(path)!=manifest[split+'_sha256']:raise RuntimeError('data changed')
        splits[split]=rows(path)
    a.output.mkdir(parents=True,exist_ok=False)
    receipt={'selection':selection,'selection_sha256':sha(a.training/'selection.json'),
             'opened_at':datetime.now(timezone.utc).isoformat(),'data_manifest_sha256':sha(a.data/'manifest.json'),
             'model_sha256':sha(a.model),'executable_sha256':sha('artifacts/bin/readout'),
             'code_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    (a.output/'evaluation-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    shuffled=[]
    for r in splits['test']:
        if r['id'] not in manifest['shuffle_ids']:continue
        r=copy.deepcopy(r);options=list(r['question']['criteria'].items());original=list(options)
        random.Random(manifest['seed']+'|shuffle|'+r['id']).shuffle(options)
        if options==original:options=options[1:]+options[:1]
        r['question']['criteria']=dict(options);shuffled.append(r)
    summary={'receipt':receipt,'models':{}}
    answers={}
    for name,adapter in (('base',None),('adapter',selected['native'])):
        dest=a.output/name;dest.mkdir();before=gpu_free_mib();backend=NativeReadout('artifacts/bin/readout',a.model,16384,adapter=adapter)
        engine=DecisionEngine(backend);minimum=before
        def infer(records,phase):
            nonlocal minimum
            predictions={}
            with (dest/(phase+'.jsonl')).open('x') as f:
                for i,r in enumerate(records):
                    start=time.monotonic()
                    item={'id':r['id'],'input_sha256':r['input_sha256'],'error':None}
                    try:
                        answer,traces=engine.answer(r['state'],r['question']);item.update(answer=answer,traces=traces)
                    except ValueError as e:item['error']=str(e)
                    item['latency_ms']=(time.monotonic()-start)*1000
                    minimum=min(minimum,gpu_free_mib());predictions[r['id']]=item
                    f.write(json.dumps(item,ensure_ascii=False)+'\n');f.flush()
                    if (i+1)%100==0:print(name,phase,i+1,flush=True)
            return predictions
        try:
            native_dev=infer(splits['dev'],'dev')
            step=0 if name=='base' else selected['update']
            torch_dev=json.loads((a.training/f'dev-predictions-{step:04}.json').read_text())
            tvds=[];agrees=[]
            for r in splits['dev']:
                x=probabilities(r,native_dev[r['id']]['answer']);y=probabilities(r,torch_dev[r['id']]['answer'])
                tvds.append(sum(abs(x[k]-y[k]) for k in x)/2)
                agrees.append(max(x,key=x.__getitem__)==max(y,key=y.__getitem__))
            dev_parity={'n':len(tvds),'mean_tvd':sum(tvds)/len(tvds),'max_tvd':max(tvds),'argmax_agreement':sum(agrees)/len(agrees)}
            (dest/'trained-native-parity.json').write_text(json.dumps(dev_parity,indent=2)+'\n')
            if dev_parity['mean_tvd']>.03 or dev_parity['argmax_agreement']<.95:raise RuntimeError('trained adapter deployment parity failed before test')
            calibration_predictions=infer(splits['calibration'],'calibration')
            calibration=fit(splits['calibration'],calibration_predictions)
            (dest/'calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
            test=infer(splits['test'],'test');shuffle=infer(shuffled,'shuffle')
        finally:backend.close()
        c=Calibration(**calibration['parameters'])
        def calibrated(records,predictions):
            return {r['id']:({'answer':replay(r,predictions[r['id']],c),'latency_ms':predictions[r['id']]['latency_ms']}
                             if not predictions[r['id']]['error'] else predictions[r['id']]) for r in records}
        fitted=calibrated(splits['test'],test);fitted_shuffle=calibrated(shuffled,shuffle)
        pairs=[]
        for r in shuffled:
            x,y=fitted[r['id']],fitted_shuffle[r['id']]
            if x.get('error') or y.get('error'):
                pairs.append({'id':r['id'],'error':'incomplete pair'});continue
            x,y=x['answer'],y['answer'];keys=x['probabilities']
            pairs.append({'id':r['id'],'options':len(keys),'flip':x['choice']!=y['choice'],
                          'tvd':sum(abs(x['probabilities'][k]-y['probabilities'][k]) for k in keys)/2})
        valid=[p for p in pairs if not p.get('error')];flips=sum(p['flip'] for p in valid)
        held=[r for r in splits['test'] if r['source'] in manifest['held_out_sources']]
        metrics={'trained_native_parity':dev_parity,'raw':report(splits['test'],test),'calibrated':report(splits['test'],fitted),
                 'held_out_sources_calibrated':report(held,fitted),'calibration':calibration,
                 'order':{'planned_pairs':len(pairs),'valid_pairs':len(valid),'flips':flips,'rate':flips/len(valid),
                          'wilson_95':wilson(flips,len(valid)),'mean_tvd':sum(p['tvd'] for p in valid)/len(valid)},
                 'memory':{'before_mib':before,'minimum_sampled_free_mib':minimum,'allocation_delta_mib':before-minimum},
                 'adapter_bytes':Path(adapter).stat().st_size if adapter else 0}
        (dest/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');(dest/'order-pairs.json').write_text(json.dumps(pairs,indent=2)+'\n')
        summary['models'][name]=metrics;answers[name]=fitted
    a_metrics,b_metrics=[summary['models'][n]['calibrated']['overall'] for n in ('base','adapter')]
    summary['accuracy_delta_percentage_points']=100*(b_metrics['accuracy_failures_incorrect']-a_metrics['accuracy_failures_incorrect'])
    summary['order_flip_delta_percentage_points']=100*(summary['models']['adapter']['order']['rate']-summary['models']['base']['order']['rate'])
    summary['selection_is_unchanged_base']=selected['update']==0
    summary['limitations']=['This is one bounded adaptation, not a release-quality claim.','Test excluded from checkpoint selection; calibration used only its separate validation records.','Source holdouts do not establish absence from base pretraining.','Runtime memory samples are not a continuous peak measurement.','All speed measurements use the RTX PRO 6000, not the 4090 or a Mac.','No hosted Jev calls or claims based on new Jev measurements.']
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k not in ('receipt','models')},indent=2))

if __name__=='__main__':main()
