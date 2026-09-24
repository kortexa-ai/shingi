"""One bounded decision-LoRA run; base immutable, dev-only checkpoint choice."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
from shingi.decision import DecisionEngine
from shingi.metrics import report
from shingi.native_tokenizer import NativeTokenizer
from shingi.ternary_training import BASE_SHA256, load_bonsai, attach_lora
from training_canary import export_adapter
from prepare_training_data import APACHE_SOURCES
from shingi.sources_v3 import FIT, SYNTHETIC


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def rows(path):
    with Path(path).open() as f:return [json.loads(line) for line in f if line.strip()]


def validate_fitting_sources(manifest, prepared, dev):
    if manifest.get('profile', '').startswith('v3-'):
        allowed = set(FIT) | set(SYNTHETIC)
        if set(manifest['train_sources']) - allowed or any(r['source'] not in allowed for r in prepared + dev):
            raise ValueError('non-allowlisted data v3 fitting source')
        return
    if manifest.get('profile') != 'apache-v2': return
    if set(manifest['train_sources']) != set(APACHE_SOURCES):
        raise ValueError('Apache profile requires the exact audited source allowlist')
    if any(r['source'] not in APACHE_SOURCES for r in prepared + dev):
        raise ValueError('non-allowlisted fitting source')


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--prism',type=Path,required=True)
    p.add_argument('--canary',type=Path,required=True);p.add_argument('--data',type=Path,default=Path('artifacts/decision-v2'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--max-seconds',type=int,default=21600)
    p.add_argument('--eval-every',type=int,default=128)
    a=p.parse_args()
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('source must be committed')
    canary=json.loads(a.canary.read_text())
    context=json.loads((a.data/'tokenized-manifest.json').read_text())['max_tokens']
    if not canary.get('passed') or canary['base_sha256']!=BASE_SHA256 or canary.get('backward_context_tokens')!=context:
        raise RuntimeError(f'successful matching {context}-token parity/backward canary required')
    if sha(a.model)!=BASE_SHA256:raise RuntimeError('base checksum mismatch')
    a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((a.data/'manifest.json').read_text())
    for split in ('train-prompts','dev'):
        if sha(a.data/(split+'.jsonl'))!=manifest[split+'_sha256']:raise RuntimeError('dataset hash mismatch')
    token_manifest=json.loads((a.data/'tokenized-manifest.json').read_text())
    if token_manifest['data_manifest_sha256']!=sha(a.data/'manifest.json') or token_manifest['tokens_sha256']!=sha(a.data/'tokenized.jsonl'):
        raise RuntimeError('prepared tokenization provenance differs')
    prepared=rows(a.data/'tokenized.jsonl')
    excluded=token_manifest['excluded']
    tokenizer=NativeTokenizer('artifacts/bin/readout',a.model)
    dev=rows(a.data/'dev.jsonl')
    validate_fitting_sources(manifest, prepared, dev)
    apache = manifest.get('profile') == 'apache-v2'
    v3 = manifest.get('profile', '').startswith('v3-')
    canonical = apache or v3
    if not prepared:raise RuntimeError('no training records')
    (a.output/'excluded-length.json').write_text(json.dumps(excluded,indent=2)+'\n')
    import torch
    from safetensors.torch import save_file
    from shingi import gpu as rails
    rails.require_training_gpu()
    torch.set_num_threads(12);torch.manual_seed(20260924 if v3 else 20260923 if apache else 20260922)
    started=time.monotonic();stop=False
    def request_stop(*unused):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,request_stop)
    model,info=load_bonsai(a.model,a.prism)
    params=attach_lora(model,rank=8,alpha=16,last_layers=64)
    optimizer=torch.optim.AdamW(params,lr=5e-5,betas=(.9,.95),weight_decay=.01,eps=1e-8)
    scaler=torch.amp.GradScaler('cuda',init_scale=1.0,growth_interval=1000000)
    accumulation=8;updates=math.ceil(len(prepared)/accumulation)
    if apache: updates = min(updates, 640)
    provenance={'code_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'model_sha256':BASE_SHA256,'canary_sha256':sha(a.canary),'dataset_manifest_sha256':sha(a.data/'manifest.json'),
                'trainable_parameters':sum(p.numel() for p in params),'prepared_prompts':len(prepared),'excluded_prompts':len(excluded),
                'train_by_source':dict(Counter(r['source'] for r in prepared)), 'rank':8,'alpha':16,'learning_rate':5e-5,
                'gradient_accumulation':accumulation,'max_context':context,'eval_every':a.eval_every,'max_seconds':a.max_seconds,
                'planned_updates':updates,'loss':'candidate cross entropy against human distribution when available, otherwise gold label',
                'checkpoint_selection':'lowest uncalibrated development NLL; test and calibration not read during training',
                'initialization':'fresh zero-output LoRA; immutable base; no previous adapter or optimizer',
                'choice_order':'canonical' if canonical else 'input',
                'profile':manifest.get('profile', 'original-v1'),
                'early_stop_patience':2 if canonical else None,
                'loader':info,'checkpoints':[],'skipped_optimizer_steps':0}
    def save_run():
        (a.output/'run.json').write_text(json.dumps(provenance,indent=2)+'\n')
    def headroom():
        torch.cuda.synchronize();rails.assert_post_first_backward_free()
        return {'free_gib':torch.cuda.mem_get_info()[0]/2**30,'allocated_gib':torch.cuda.memory_allocated()/2**30,
                'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30}
    class Backend:
        def infer(self,prompt,labels):
            r=tokenizer.encode(prompt,labels)
            if r['input_tokens']>8192:raise ValueError('development prompt exceeds 8192-token inference bound')
            t=time.monotonic()
            ids=torch.tensor([r['input_ids']],device='cuda')
            h=model.model(input_ids=ids,use_cache=False).last_hidden_state[:,-1:,:]
            out=model.lm_head(h)[0,0,r['candidate_ids']].float().cpu().tolist()
            rails.assert_post_first_backward_free()
            return {'logits':out,'input_tokens':r['input_tokens'],'candidate_ids':r['candidate_ids'],
                    'prefill_ms':1000*(time.monotonic()-t),'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()}
    backend=Backend();engine=DecisionEngine(backend, canonical_choices=canonical)
    best_nll=float('inf');best_path=None
    def evaluate(step):
        nonlocal best_nll,best_path
        model.eval();model.gradient_checkpointing_disable();predictions={}
        with torch.no_grad():
            for r in dev:
                answer,traces=engine.answer(r['state'],r['question'])
                predictions[r['id']]={'answer':answer,'traces':traces}
        with (a.output/f'dev-predictions-{step:04}.json').open('x') as f:json.dump(predictions,f)
        metrics=report(dev,predictions);nll=metrics['overall']['nll']['mean']
        (a.output/f'dev-{step:04}.json').write_text(json.dumps(metrics,indent=2)+'\n')
        if not math.isfinite(nll) or metrics['overall']['valid']!=len(dev):raise RuntimeError('invalid development evaluation')
        state={k:v.detach().cpu().contiguous() for k,v in model.named_parameters() if v.requires_grad}
        checkpoint=a.output/f'adapter-{step:04}.safetensors';save_file(state,str(checkpoint))
        native=a.output/f'adapter-{step:04}.gguf';export_adapter(model,native,a.prism)
        record={'update':step,'dev_nll':nll,'dev_accuracy':metrics['overall']['accuracy_failures_incorrect'],
                'checkpoint':str(checkpoint),'sha256':sha(checkpoint),'native':str(native),'native_sha256':sha(native),
                'elapsed_seconds':time.monotonic()-started,'memory':headroom()}
        provenance['checkpoints'].append(record)
        if nll<best_nll:
            best_nll=nll;best_path=record
        provenance['selected']=best_path;save_run();print(json.dumps(record),flush=True)
        model.train();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    evaluate(0)
    trained=0;loss_sum=0.;last_eval=0;stale_evaluations=0
    try:
        with (a.output/'steps.jsonl').open('x') as log:
            for step in range(updates):
                if stop or time.monotonic()-started>=a.max_seconds:break
                group=prepared[step*accumulation:(step+1)*accumulation]
                warmup=min(1.,(step+1)/32)
                decay=.2+.8*.5*(1+math.cos(math.pi*step/max(1,updates-1)))
                lr=5e-5*warmup*decay
                for param_group in optimizer.param_groups:param_group['lr']=lr
                before=time.monotonic();optimizer.zero_grad(set_to_none=True);group_loss=0.
                for item in group:
                    ids=torch.tensor([item['input_ids']],device='cuda')
                    h=model.model(input_ids=ids,use_cache=False).last_hidden_state[:,-1:,:]
                    logits=model.lm_head(h)[0,0,item['candidate_ids']].float()
                    target=torch.tensor(item['target'],device='cuda',dtype=torch.float32)
                    loss=-(target*torch.log_softmax(logits,dim=-1)).sum()
                    if not torch.isfinite(loss):raise RuntimeError('non-finite training loss')
                    scaler.scale(loss/len(group)).backward();group_loss+=loss.item()
                    if trained==0:provenance['first_backward']=headroom();save_run()
                    trained+=1
                scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(params,1.)
                finite=torch.isfinite(norm).item();scale_before=scaler.get_scale()
                if finite:
                    scaler.step(optimizer);scaler.update()
                else:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update(new_scale=scale_before/2)
                if not finite:
                    provenance['skipped_optimizer_steps']+=1
                    if provenance['skipped_optimizer_steps']>10:raise RuntimeError('too many non-finite-gradient steps')
                memory=headroom()
                record={'update':step+1,'examples':trained,'loss':group_loss/len(group),'gradient_norm':norm.item() if finite else None,
                        'optimizer_step_skipped':not finite,'loss_scale':scaler.get_scale(),'learning_rate':lr,
                        'seconds':time.monotonic()-before,'memory':memory}
                log.write(json.dumps(record)+'\n');log.flush()
                if (step+1)%16==0:print(json.dumps(record),flush=True)
                provenance['completed_updates']=step+1;provenance['processed_prompts']=trained
                if (step+1)%a.eval_every==0:
                    previous_best = best_nll
                    evaluate(step+1);last_eval=step+1
                    stale_evaluations = stale_evaluations+1 if best_nll >= previous_best else 0
                    if canonical and stale_evaluations >= 2:
                        provenance['stop_reason']='two development evaluations without improvement'
                        break
            completed=provenance.get('completed_updates',0)
            if completed!=last_eval:evaluate(completed)
        provenance['status']='finished' if completed==updates else 'bounded-stop'
        provenance['elapsed_seconds']=time.monotonic()-started;save_run()
        # Immutable selection receipt before any calibration or test inference.
        baseline = provenance['checkpoints'][0]
        accepted = (best_path['update'] > 0 and best_path['dev_nll'] < baseline['dev_nll']
                    and best_path['dev_accuracy'] >= baseline['dev_accuracy']-.02)
        (a.output/'selection.json').write_text(json.dumps({'selected':best_path,'reason':'minimum development NLL',
            'test_seen':False, 'development_gate_passed':accepted,
            'gate':'NLL improves unchanged base and accuracy loses at most 2 percentage points'},indent=2)+'\n')
    except BaseException as exc:
        provenance['status']='failed';provenance['failure']=f'{type(exc).__name__}: {exc}';save_run();raise
    finally:tokenizer.close()

if __name__=='__main__':main()
