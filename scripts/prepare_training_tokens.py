"""Validate and freeze native tokenization on CPU before production downtime."""
import argparse
import json
from pathlib import Path
from shingi.native_tokenizer import NativeTokenizer
from train_decision_adapter import rows,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--data',type=Path,default=Path('artifacts/decision-v2'))
    p.add_argument('--max-tokens',type=int,default=1024);a=p.parse_args()
    manifest=json.loads((a.data/'manifest.json').read_text())
    if manifest['train-prompts_sha256']!=sha(a.data/'train-prompts.jsonl'):raise RuntimeError('training prompts changed')
    tokenizer=NativeTokenizer('artifacts/bin/readout',a.model);excluded=[];count=0;tokens={'synthetic':0,'natural':0}
    try:
        with (a.data/'tokenized.jsonl').open('x') as f:
            for row in rows(a.data/'train-prompts.jsonl'):
                encoded=tokenizer.encode(row['prompt'],row['labels'])
                if encoded['input_tokens']>a.max_tokens:
                    excluded.append({'id':row['id'],'source':row['source'],'tokens':encoded['input_tokens'],'reason':'exceeds context; not truncated'})
                    continue
                f.write(json.dumps({**row,**encoded},ensure_ascii=False)+'\n');count+=1
                tokens['synthetic' if row['source'].startswith('synth_') else 'natural']+=encoded['input_tokens']
    finally:tokenizer.close()
    result={'data_manifest_sha256':sha(a.data/'manifest.json'),'tokens_sha256':sha(a.data/'tokenized.jsonl'),
            'executable_sha256':sha('artifacts/bin/readout'),'max_tokens':a.max_tokens,'prepared':count,'excluded':excluded,
            'input_tokens':tokens,'synthetic_token_share':tokens['synthetic']/max(1,sum(tokens.values()))}
    # Data v3 caps synthetic prompts at 40% of training tokens.
    if manifest.get('profile','').startswith('v3-') and result['synthetic_token_share']>.40:
        raise RuntimeError(f"synthetic token share {result['synthetic_token_share']:.3f} exceeds 0.40")
    with (a.data/'tokenized-manifest.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'prepared':count,'excluded':len(excluded),'synthetic_token_share':result['synthetic_token_share'],'data_manifest_sha256':result['data_manifest_sha256']}))

if __name__=='__main__':main()
