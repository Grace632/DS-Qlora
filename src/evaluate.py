"""Run a fixed deterministic Base-vs-adapter evaluation and exact quote audit."""
import argparse
import json
import re
import time
from pathlib import Path

import torch
from peft import PeftModel
from test_model import load


def normalize(text):
    return re.sub(r'\s+', '', text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/DeepSeek-V2-Lite-Chat')
    parser.add_argument('--adapter', default='checkpoints/adapter_formal')
    parser.add_argument('--questions', default='data/evaluation/test_questions.json')
    parser.add_argument('--output', default='outputs/evaluation/base_vs_formal.json')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    args = parser.parse_args()
    questions=json.loads(Path(args.questions).read_text('utf-8'))
    assert len(questions)>=50 and len({x['question'] for x in questions})==len(questions)
    sources=Path('data/raw/cbeta')
    originals='\n'.join(p.read_text('utf-8-sig') for p in sources.glob('T*.txt'))
    original_norm=normalize(originals)
    tokenizer, base=load(args.model)
    model=PeftModel.from_pretrained(base,args.adapter,is_trainable=False)
    model.eval()
    results=[]
    start=time.monotonic()
    for index,row in enumerate(questions,1):
        prompt=tokenizer.apply_chat_template([{'role':'user','content':row['question']}],
            tokenize=True,add_generation_prompt=True,return_tensors='pt',return_dict=True)
        prompt={k:v.to('cuda') for k,v in prompt.items()}
        pair={}
        for mode in ('base','qlora'):
            context=model.disable_adapter() if mode=='base' else __import__('contextlib').nullcontext()
            with context,torch.inference_mode():
                generated=model.generate(**prompt,max_new_tokens=args.max_new_tokens,
                    do_sample=False,repetition_penalty=1.08,use_cache=True,
                    pad_token_id=tokenizer.eos_token_id)
            answer=tokenizer.decode(generated[0,prompt['input_ids'].shape[1]:],skip_special_tokens=True).strip()
            quotes=re.findall(r'[「『\"“]([^」』\"”]{8,})[」』\"”]',answer)
            audited=[{'text':q,'exact_source_match':normalize(q) in original_norm}
                     for q in quotes if re.search(r'[\u3400-\u9fff]{4}',q)]
            citation_mentioned=bool(re.search(r'《[^》]{1,32}》|【經文依據】|【经典依据】|[「『“]',answer))
            pair[mode]={'answer':answer,'quoted_spans':audited,'citation_mentioned':citation_mentioned}
        results.append({**row,'models':pair})
        if index%5==0:
            print(json.dumps({'evaluated':index,'total':len(questions),'elapsed_seconds':round(time.monotonic()-start,1)}),flush=True)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'generation':{'max_new_tokens':args.max_new_tokens,'do_sample':False,
        'repetition_penalty':1.08,'quantization':'4-bit NF4 double quant, BF16 compute'},
        'independent_questions':len(questions),'elapsed_seconds':round(time.monotonic()-start,1),
        'quote_audit_rule':'quoted substring with >=4 CJK characters; whitespace normalized; exact substring searched in all eight downloaded CBETA originals; automatic lexical audit only, not doctrinal review',
        'results':results},ensure_ascii=False,indent=2)+'\n','utf-8')
    print('SAVED',str(output),'questions',len(results),'seconds',round(time.monotonic()-start,1),flush=True)


if __name__=='__main__': main()
