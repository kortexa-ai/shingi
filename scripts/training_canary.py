"""Native -> differentiable Bonsai -> native adapter parity and memory canary."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from shingi.backend import NativeReadout
from shingi.decision import prompt_for, LETTERS
from shingi.ternary_training import BASE_SHA256, load_bonsai, attach_lora, hadamard


def export_adapter(model, path, prism_root, alpha=16):
    sys.path.insert(0, str(prism_root / 'gguf-py'))
    from gguf import GGUFWriter
    w = GGUFWriter(str(path), 'qwen35')
    w.add_string('general.type', 'adapter')
    w.add_string('adapter.type', 'lora')
    w.add_float32('adapter.lora.alpha', alpha)
    mapping = {'gate_proj':'ffn_gate', 'up_proj':'ffn_up', 'down_proj':'ffn_down'}
    for i, layer in enumerate(model.model.layers):
        for name, native in mapping.items():
            module = getattr(layer.mlp, name)
            if not hasattr(module, 'a'):
                continue
            for attr, suffix in [('a','lora_a'),('b','lora_b')]:
                # Native LoRA consumes the ORIGINAL activation, before Hadamard.
                # Only MLP projections are trained, so no GDN head permutation applies.
                array = getattr(module, attr).detach().cpu().float().numpy()
                w.add_tensor(f'blk.{i}.{native}.weight.{suffix}', array)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def make_prompts():
    examples = [
        ('The invoice is due tomorrow and has not been paid.', 'What action fits?', [('pay','Pay the invoice'),('ignore','Discard the invoice'),('archive','Archive as paid')]),
        ('The passenger asks to cancel a hotel reservation.', 'What does the passenger want?', [('flight','Book a flight'),('cancel','Cancel a hotel reservation'),('food','Order food')]),
        ('A red ball is inside a blue box.', 'What color is the ball?', [('red','Red'),('blue','Blue'),('green','Green')]),
        ('No messages arrived today.', 'Did any messages arrive?', [('yes','The statement is true.'),('no','The statement is false.')]),
        ('The service was excellent and I will return.', 'Rate the sentiment.', [('0','Very negative'),('1','Negative'),('2','Neutral'),('3','Positive'),('4','Very positive')]),
        ('7 multiplied by 8 is 56.', 'Which result is correct?', [('a','54'),('b','56'),('c','63'),('d','72')]),
    ]
    rows = []
    for state, question, options in examples:
        for order in (options, list(reversed(options))):
            rows.append({'prompt':prompt_for(state,question,order),'labels':list(LETTERS[:len(order)])})
    for n in (8, 16, 32, 52):
        options = [(str(i), f'Value {i}') for i in range(n)]
        rows.append({'prompt':prompt_for(f'The stored value is {n-2}.','Select the stored value.',options),'labels':list(LETTERS[:n])})
    return rows


def compare(actual, expected):
    from shingi.decision import softmax
    diffs, tvds, agrees = [], [], []
    for a,b in zip(actual,expected):
        a,b = np.array(a),np.array(b)
        diffs.extend(((a-a.mean())-(b-b.mean())).tolist())
        tvds.append(float(np.abs(np.array(softmax(a.tolist())) - np.array(softmax(b.tolist()))).sum()/2))
        agrees.append(int(a.argmax()==b.argmax()))
    return {'centered_logit_rmse':float(np.sqrt(np.mean(np.square(diffs)))), 'mean_tvd':float(np.mean(tvds)),
            'max_tvd':max(tvds),'argmax_agreement':float(np.mean(agrees)), 'prompts':len(agrees)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--prism',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-a71210ca-e14a-755a-88bb-77f53a2102f6': raise RuntimeError('6000 pin required')
    digest=hashlib.file_digest(a.model.open('rb'),'sha256').hexdigest()
    if digest!=BASE_SHA256: raise ValueError('base checksum mismatch')
    os.environ['SHINGI_KV_F16']='1'
    rows=make_prompts();native=NativeReadout('artifacts/bin/readout',a.model,2048)
    try:
        for r in rows:
            r['tokenized']=native.infer(r['prompt'],r['labels'],tokenize_only=True)
            r['native']=native.infer(r['prompt'],r['labels'])['logits']
    finally: native.close()
    (a.output/'native-reference.json').write_text(json.dumps(rows,indent=2))
    import torch
    sys.path.insert(0,'/home/francip/src/legolm/scripts')
    import smarty_gpu_rails as rails
    torch.set_num_threads(12);torch.manual_seed(42)
    x=torch.randn(3,2048);sign=torch.randint(0,2,(2048,))*2-1
    torch.testing.assert_close(hadamard(hadamard(x,sign),sign,inverse=True),x,atol=1e-6,rtol=1e-5)
    result={'base_sha256':digest,'native_kv':'f16','stages':{}}
    def memory(stage):
        torch.cuda.synchronize()
        result['stages'][stage]={'free_gib':torch.cuda.mem_get_info()[0]/2**30,'allocated_gib':torch.cuda.memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30}
        rails.assert_post_first_backward_free();print(stage,result['stages'][stage],flush=True)
        (a.output/'parity.json').write_text(json.dumps(result,indent=2)+'\n')
    memory('before_load')
    model,info=load_bonsai(a.model,a.prism);result['loader']=info;memory('after_load')
    def forward(r):
        ids=torch.tensor([r['tokenized']['input_ids']],device='cuda')
        h=model.model(input_ids=ids,use_cache=False).last_hidden_state[:,-1:,:]
        return model.lm_head(h)[0,0,r['tokenized']['candidate_ids']].float()
    started=time.monotonic()
    with torch.no_grad(): actual=[forward(r).cpu().tolist() for r in rows]
    result['base_parity']=compare(actual,[r['native'] for r in rows]);result['forward_seconds']=time.monotonic()-started
    (a.output/'parity.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result['base_parity']),flush=True)
    m=result['base_parity']
    if m['argmax_agreement']<.95 or m['mean_tvd']>.03 or m['max_tvd']>.10 or m['centered_logit_rmse']>.30:
        raise RuntimeError('base parity gate failed; no training permitted')
    params=attach_lora(model);result['trainable_parameters']=sum(p.numel() for p in params)
    with torch.no_grad(): zero=forward(rows[0]).cpu().tolist()
    np.testing.assert_array_equal(zero,actual[0])
    result['zero_adapter_exact']=True
    model.train();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    optimizer=torch.optim.AdamW(params,lr=2e-5);scaler=torch.amp.GradScaler('cuda', init_scale=1.0)
    # Synthetic canary only: this gradient is never reused as a trained checkpoint.
    start=time.monotonic();logits=forward(rows[0]);loss=torch.nn.functional.cross_entropy(logits[None],torch.tensor([1],device='cuda'))
    scaler.scale(loss).backward();memory('first_backward')
    scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(params,1.0)
    if not torch.isfinite(norm): raise RuntimeError('non-finite gradients')
    scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);memory('after_optimizer')
    result['loss_scale']=scaler.get_scale();result['step_seconds']=time.monotonic()-start;result['loss']=loss.item();result['gradient_norm']=norm.item()
    model.eval();model.gradient_checkpointing_disable()
    with torch.no_grad(): trained=[forward(r).cpu().tolist() for r in rows]
    export_adapter(model,a.output/'canary.gguf',a.prism)
    del model,params,optimizer,scaler,logits,loss;gc.collect();torch.cuda.empty_cache()
    native=NativeReadout('artifacts/bin/readout',a.model,2048,adapter=a.output/'canary.gguf')
    try: reloaded=[native.infer(r['prompt'],r['labels'])['logits'] for r in rows]
    finally: native.close()
    result['adapter_parity']=compare(trained,reloaded)
    result['canary_adapter_not_for_training']=True
    m=result['adapter_parity'];result['passed']=m['argmax_agreement']>=.95 and m['mean_tvd']<=.03 and m['max_tvd']<=.10 and m['centered_logit_rmse']<=.30
    (a.output/'parity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
    if not result['passed']: raise RuntimeError('native adapter parity failed')

if __name__=='__main__':main()
