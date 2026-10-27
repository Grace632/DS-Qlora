"""Bounded LLaMA-Factory QLoRA with source data checks and in-memory audits."""
import argparse
import contextlib
import copy
import io
import json
import math
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import bitsandbytes as bnb
import torch
import yaml
from transformers import TrainerCallback
from llamafactory.hparams import get_train_args
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer, SFTDataCollatorWith4DAttentionMask
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from llamafactory.data.template import register_template
from llamafactory.data.formatter import StringFormatter, EmptyFormatter

# Official tokenizer generates 'Assistant:' as prompt and starts the answer
# with a space. The built-in deepseek template omits that answer-leading space.
register_template(
    name='deepseek_v2_lite_official',
    format_user=StringFormatter(slots=['User: {{content}}\n\nAssistant:']),
    format_assistant=StringFormatter(slots=[' {{content}}', {'eos_token'}]),
    format_system=StringFormatter(slots=['{{content}}\n\n']),
    format_prefix=EmptyFormatter(slots=[{'bos_token'}]),
)


class VerifyUpdates(TrainerCallback):
    def __init__(self):
        self.steps = self.grad_steps = self.update_steps = 0
        self.before = None
        self.losses, self.validation, self.seconds = [], [], []

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        assert model.is_loaded_in_4bit
        quant=model.config.quantization_config
        assert quant.bnb_4bit_quant_type == 'nf4' and quant.bnb_4bit_use_double_quant
        assert quant.bnb_4bit_compute_dtype == torch.bfloat16
        assert all(p.device.type=='cuda' for p in model.parameters()), 'CPU offload is not allowed'
        self.params=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        assert self.params and all('lora_' in n for n,_ in self.params)
        self.base=[(n,p) for n,p in model.named_parameters() if 'lora_' not in n]
        assert all(not p.requires_grad for _,p in self.base)
        # LLaMA-Factory stores packed 4-bit bytes in BF16 tensors. Some packed
        # bit patterns represent floating-point NaN; compare bytes, not values.
        self.base_samples=[(n,p,p.detach().view(torch.uint8).flatten()[:128].clone()) for n,p in self.base]
        packed_nan=sum(bool(torch.isnan(p.detach().flatten()[:64]).any().item())
                       for _,p in self.base if isinstance(p,bnb.nn.Params4bit) and p.is_floating_point())
        count=sum(p.numel() for _,p in self.params)
        total=model.get_nb_trainable_parameters()[1]
        self.started=self.last=time.monotonic()
        print('TRAIN_BEGIN',json.dumps({'trainable_parameters':count,'total_parameters':total,
            'trainable_percent':100*count/total,'lora_tensors':len(self.params),
            'nf4_layers':sum(isinstance(m,bnb.nn.Linear4bit) for m in model.modules()),
            'base_frozen':True,'gradient_checkpointing':model.is_gradient_checkpointing,
            'packed_storage_samples_with_nan_bitpatterns':packed_nan,
            'gradient_accumulation_steps':args.gradient_accumulation_steps}),flush=True)

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        grads=[p.grad for _,p in self.params if p.grad is not None]
        assert grads and torch.stack([torch.isfinite(g).all() for g in grads]).all().item()
        norm=torch.stack([g.float().square().sum() for g in grads]).sum().sqrt().item()
        assert math.isfinite(norm) and norm>0, 'LoRA gradients must be nonzero'
        assert all(p.grad is None and not p.requires_grad for _,p in self.base)
        self.grad_steps+=1
        self.before=[p.detach().clone() for _,p in self.params]
        self.norm=norm

    def on_step_end(self, args, state, control, **kwargs):
        self.steps=state.global_step
        changes=torch.stack([(p.detach()-v).abs().max() for v,(_,p) in zip(self.before,self.params)])
        delta=changes.max().item()
        assert math.isfinite(delta) and delta>0, 'LoRA optimizer did not update weights'
        self.update_steps+=1
        self.before=None
        now=time.monotonic(); self.seconds.append(now-self.last); self.last=now
        print('OPTIMIZER_STEP',json.dumps({'step':self.steps,'grad_norm_after_clip':self.norm,
            'max_parameter_delta':delta,'seconds':round(self.seconds[-1],2),
            'allocated_mib':round(torch.cuda.memory_allocated()/2**20,1)}),flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            value=float(logs['loss']); assert math.isfinite(value)
            self.losses.append({'step':state.global_step,'loss':value})
        if logs and 'eval_loss' in logs:
            value=float(logs['eval_loss']); assert math.isfinite(value)
            self.validation.append({'step':state.global_step,'loss':value})

    def on_train_end(self, args, state, control, **kwargs):
        assert self.steps>=10 and self.grad_steps==self.steps and self.update_steps==self.steps
        assert self.losses and all(math.isfinite(x['loss']) for x in self.losses)
        assert self.validation
        unchanged=torch.stack([torch.eq(p.detach().view(torch.uint8).flatten()[:128],v).all()
                               for _,p,v in self.base_samples]).all().item()
        assert unchanged and all(p.grad is None and not p.requires_grad for _,p in self.base)
        self.result={'optimizer_steps':self.steps,'nonzero_gradient_steps':self.grad_steps,
            'parameter_update_steps':self.update_steps,'base_frozen_and_sampled_values_unchanged':True,
            'losses':self.losses,'validation_losses':self.validation,'training_seconds':round(time.monotonic()-self.started,2),
            'step_seconds':[round(x,2) for x in self.seconds],
            'peak_allocated_mib':round(torch.cuda.max_memory_allocated()/2**20,1),
            'peak_reserved_mib':round(torch.cuda.max_memory_reserved()/2**20,1)}
        print('VERIFIED',json.dumps(self.result),flush=True)
        self.before=None; self.base_samples=[]


class CheckedTrainer(CustomSeq2SeqTrainer):
    """Reject invalid raw losses before Trainer can hide NaN/Inf in logging."""
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        result=super().compute_loss(model,inputs,return_outputs=return_outputs,**kwargs)
        loss=result[0] if return_outputs else result
        assert torch.isfinite(loss.detach()).all().item(), 'non-finite microbatch loss'
        if model.training:
            self.checked_microbatches=getattr(self,'checked_microbatches',0)+1
        return result


def checked_datasets(ma, da, ta, tm, template):
    loaded={}
    for split in ('train','validation'):
        data_args=copy.deepcopy(da); data_args.dataset=[f'buddha_qa_{split}']; data_args.eval_dataset=None
        with contextlib.redirect_stdout(io.StringIO()):
            module=get_dataset(template,ma,data_args,ta,stage='sft',**tm)
        data=module['train_dataset']
        rows=json.loads((Path(da.dataset_dir)/f'sft_{split}.json').read_text('utf-8'))
        assert len(rows)==len(data)
        lengths=[]
        for raw,item in zip(rows,data):
            messages=raw['messages']
            assert [m['role'] for m in messages][-2:]==['user','assistant']
            assert all(m['content'].strip() for m in messages)
            full=tm['tokenizer'].apply_chat_template(messages,tokenize=True,add_generation_prompt=False)
            prompt=tm['tokenizer'].apply_chat_template(messages[:1],tokenize=True,add_generation_prompt=True)
            assert full==item['input_ids'], ('template mismatch or truncated sample',raw['id'])
            assert len(full)<=da.cutoff_len
            assert item['labels'][:len(prompt)]==[-100]*len(prompt), ('unmasked prompt',raw['id'])
            assert item['labels'][len(prompt):]==full[len(prompt):], ('answer mask mismatch',raw['id'])
            lengths.append(len(full))
        print('DATA_CHECK',json.dumps({'split':split,'rows':len(data),'min_tokens':min(lengths),
            'max_tokens':max(lengths),'mean_tokens':round(sum(lengths)/len(lengths),1),
            'no_truncation':True,'native_template_match':True,'assistant_only_labels':True}),flush=True)
        loaded[split]=data
    return loaded


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='configs/qlora.yaml')
    ap.add_argument('--check-data',action='store_true')
    ap.add_argument('--check-lengths',nargs='+',type=int,help='Forward/backward only; pack real training dialogues; save no weights')
    args=ap.parse_args()
    config=yaml.safe_load(Path(args.config).read_text('utf-8'))
    assert config['quantization_method']=='bnb' and config['quantization_bit']==4
    out=Path(config['output_dir'])
    if not args.check_data and not args.check_lengths and out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Will not overwrite an existing adapter: {out}')
    with tempfile.TemporaryDirectory(prefix='buddha_qlora_') as cache:
        config['cache_dir']=cache
        if args.check_data or args.check_lengths:
            config['output_dir']=str(Path(cache)/'no_output')
        ma,da,ta,fa,ga=get_train_args(config)
        tm=load_tokenizer(ma)
        template=get_template_and_fix_tokenizer(tm['tokenizer'],da)
        data=checked_datasets(ma,da,ta,tm,template)
        if args.check_data: return
        torch.cuda.reset_peak_memory_stats()
        model=load_model(tm['tokenizer'],ma,fa,is_trainable=True)
        assert model.is_loaded_in_4bit, 'Quantization was not enabled; refuse training'
        if args.check_lengths:
            model.train()
            for length in args.check_lengths:
                assert 1<length<=2048
                ids=[]; labels=[]
                for row in data['train']:
                    ids.extend(row['input_ids']); labels.extend(row['labels'])
                    if len(ids)>=length: break
                assert len(ids)>=length
                batch={'input_ids':torch.tensor([ids[:length]],device='cuda'),
                       'labels':torch.tensor([labels[:length]],device='cuda'),
                       'attention_mask':torch.ones((1,length),dtype=torch.long,device='cuda')}
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize(); started=time.monotonic()
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                    loss=model(**batch,use_cache=False).loss
                assert torch.isfinite(loss).item()
                loss.backward(); torch.cuda.synchronize()
                gradients=[p.grad for n,p in model.named_parameters() if 'lora_' in n and p.grad is not None]
                norm=torch.stack([g.float().square().sum() for g in gradients]).sum().sqrt().item()
                assert norm>0 and math.isfinite(norm)
                assert all(not p.requires_grad and p.grad is None for n,p in model.named_parameters() if 'lora_' not in n)
                print('LENGTH_CHECK',json.dumps({'tokens':length,'loss':loss.item(),'gradient_norm':norm,
                    'seconds':round(time.monotonic()-started,2),'optimizer_steps':0,
                    'peak_allocated_mib':round(torch.cuda.max_memory_allocated()/2**20,1),
                    'peak_reserved_mib':round(torch.cuda.max_memory_reserved()/2**20,1)}),flush=True)
                model.zero_grad(set_to_none=True)
                del batch,loss,gradients
            return
        collator=SFTDataCollatorWith4DAttentionMask(template=template,model=model,
            pad_to_multiple_of=8,label_pad_token_id=-100,
            attn_implementation=model.config._attn_implementation,compute_dtype=ma.compute_dtype,**tm)
        audit=VerifyUpdates()
        trainer=CheckedTrainer(model=model,args=ta,finetuning_args=fa,model_args=ma,
            train_dataset=data['train'],eval_dataset=data['validation'],data_collator=collator,callbacks=[audit],**tm)
        result=trainer.train()
        expected=ta.max_steps if ta.max_steps>0 else int(math.ceil(len(data['train'])/ta.gradient_accumulation_steps)*ta.num_train_epochs)
        assert audit.steps==expected
        print('MICROBATCHES',trainer.checked_microbatches,'TRAIN_METRICS',json.dumps(result.metrics),flush=True)
        # Persist the training audit before model serialization so a later save
        # or connection failure cannot discard the completed training record.
        manifest={'config':{k:v for k,v in config.items() if k!='cache_dir'},
                  'training':audit.result,'trainer_metrics':result.metrics,
                  'train_rows':len(data['train']),'validation_rows':len(data['validation']),
                  'checked_microbatches':trainer.checked_microbatches,'expected_optimizer_steps':expected,
                  'adapter_saved':False}
        metrics=Path('outputs/metrics')/(out.name+'_training.json')
        metrics.parent.mkdir(parents=True,exist_ok=True)
        metrics.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n','utf-8')
        model.save_pretrained(out,safe_serialization=True,save_embedding_layers=False)
        # Only this just-created generic PEFT card is removed; keep necessary adapter files.
        (out/'README.md').unlink(missing_ok=True)
        assert (out/'adapter_config.json').is_file() and (out/'adapter_model.safetensors').stat().st_size>0
        print('ADAPTER_SAVED',[(f.name,f.stat().st_size) for f in out.iterdir()],flush=True)
        manifest['adapter_saved']=True
        metrics.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n','utf-8')


if __name__=='__main__': main()
