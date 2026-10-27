"""Offline NF4 inference and optional reloaded LoRA verification."""
import argparse
import json
import re
import time
from pathlib import Path

import bitsandbytes as bnb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False, local_files_only=True)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=False, local_files_only=True,
        quantization_config=quant, device_map={'': 0}, dtype=torch.bfloat16,
        attn_implementation='eager')
    assert model.is_loaded_in_4bit
    assert all(p.device.type == 'cuda' for p in model.parameters()), 'CPU offload is disallowed in this check'
    print('model_class', type(model).__module__, type(model).__name__, flush=True)
    print('NF4_layers', sum(isinstance(m, bnb.nn.Linear4bit) for m in model.modules()), flush=True)
    print('attention_targets', [(n, m.in_features, m.out_features)
                               for n, m in model.named_modules()
                               if n.startswith('model.layers.0.self_attn.') and isinstance(m, bnb.nn.Linear4bit)], flush=True)
    return tokenizer, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/DeepSeek-V2-Lite-Chat')
    parser.add_argument('--adapter')
    parser.add_argument('--max-new-tokens', type=int, default=96)
    parser.add_argument('--prompt', default='如何面对学习和生活中的压力？')
    args = parser.parse_args()
    start = time.monotonic()
    tok, model = load(args.model)
    if args.adapter:
        from peft import PeftModel, get_peft_model_state_dict
        from safetensors.torch import load_file
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
        saved=load_file(str(Path(args.adapter)/'adapter_model.safetensors'))
        restored=get_peft_model_state_dict(model)
        assert saved.keys()==restored.keys()
        assert all(torch.isfinite(v).all() and torch.equal(v,restored[k].detach().cpu().to(v.dtype))
                   for k,v in saved.items()), 'Adapter weights did not round-trip exactly'
        assert any(torch.count_nonzero(v).item()>0 for k,v in saved.items() if 'lora_B' in k)
        assert all(not p.requires_grad for p in model.parameters())
        print('adapter_loaded', args.adapter, 'verified_tensors',len(saved),'exact_round_trip',True,flush=True)
    model.eval()
    x = tok.apply_chat_template([{'role': 'user', 'content': args.prompt}], tokenize=True,
                               add_generation_prompt=True, return_tensors='pt', return_dict=True)
    x = {k: v.to('cuda') for k, v in x.items()}
    with torch.inference_mode():
        out = model.generate(**x, max_new_tokens=args.max_new_tokens, do_sample=False,
                             use_cache=True, pad_token_id=tok.eos_token_id)
    answer = tok.decode(out[0, x['input_ids'].shape[1]:], skip_special_tokens=True)
    assert answer.strip() and '\ufffd' not in answer, 'Invalid decoded output'
    assert re.search(r'[\u3400-\u9fff]',answer), 'Expected Chinese output'
    print('ANSWER', answer, flush=True)
    print(json.dumps(dict(elapsed_seconds=round(time.monotonic()-start, 2),
                          peak_allocated_mib=round(torch.cuda.max_memory_allocated()/2**20, 1),
                          peak_reserved_mib=round(torch.cuda.max_memory_reserved()/2**20, 1))), flush=True)


if __name__ == '__main__':
    main()
