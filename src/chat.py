"""Interactive Base/QLoRA chat using one shared 4-bit DeepSeek base."""
import argparse
import contextlib
import time

from test_model import load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/DeepSeek-V2-Lite-Chat')
    parser.add_argument('--adapter', default='checkpoints/adapter_formal')
    parser.add_argument('--mode', choices=('base', 'qlora'), default='qlora')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--max-new-tokens', type=int, default=192)
    parser.add_argument('--repetition-penalty', type=float, default=1.08)
    parser.add_argument('--max-context-tokens', type=int, default=2048)
    args = parser.parse_args()
    if args.temperature < 0 or not 0 < args.top_p <= 1 or args.max_new_tokens < 1:
        parser.error('temperature/top-p/max-new-tokens outside valid range')
    if args.repetition_penalty <= 0 or args.max_context_tokens <= args.max_new_tokens:
        parser.error('context limit must exceed generation length; repetition penalty must be positive')
    tokenizer, base = load(args.model)
    from peft import PeftModel
    adapted = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    adapted.eval()
    base.eval()
    history = []
    print('模式 base / qlora；命令 /mode base、/mode qlora、/clear、/quit')
    while True:
        try:
            question = input('\n你：').strip()
        except (EOFError, KeyboardInterrupt):
            print('\n退出。')
            break
        if not question:
            continue
        if question in ('/quit', '/exit', '退出'):
            break
        if question == '/clear':
            history.clear()
            print('对话历史已清除。')
            continue
        if question in ('/mode base', '/mode qlora'):
            args.mode = question.split()[1]
            print(f'已切换到 {args.mode}。')
            continue
        turns = history + [{'role': 'user', 'content': question}]
        inputs = tokenizer.apply_chat_template(turns, tokenize=True, add_generation_prompt=True,
                                               return_tensors='pt', return_dict=True)
        # Discard complete oldest turns rather than silently truncating a question.
        while inputs['input_ids'].shape[1] + args.max_new_tokens > args.max_context_tokens and len(turns) > 1:
            turns = turns[2:]
            inputs = tokenizer.apply_chat_template(turns, tokenize=True, add_generation_prompt=True,
                                                   return_tensors='pt', return_dict=True)
        if inputs['input_ids'].shape[1] + args.max_new_tokens > args.max_context_tokens:
            print('问题过长，请缩短后重试。')
            continue
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        sampling = args.temperature > 0
        start = time.monotonic()
        import torch
        adapter_context = adapted.disable_adapter() if args.mode == 'base' else contextlib.nullcontext()
        with adapter_context, torch.inference_mode():
            out = adapted.generate(**inputs, max_new_tokens=args.max_new_tokens,
                do_sample=sampling, temperature=args.temperature if sampling else None,
                top_p=args.top_p if sampling else None, repetition_penalty=args.repetition_penalty,
                use_cache=True, pad_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        print(f'助手（{args.mode}，{time.monotonic()-start:.1f}s）：{answer}')
        history = turns + [{'role':'assistant','content':answer}]


if __name__ == '__main__':
    main()
