# DeepSeek 佛教经典问答助手

本项目使用 DeepSeek-V2-Lite-Chat 与 4-bit NF4 QLoRA，对佛教经典问答进行小规模监督微调，并提供交互推理、Base/QLoRA 配对评测和原典字面引文核验。它是课程实验原型，不构成佛学解释或修行建议。

## 技术方案

- 基座：`deepseek-ai/DeepSeek-V2-Lite-Chat`，复用本地模型副本，`trust_remote_code=False`。
- 训练：LLaMA-Factory SFT，NF4 double quant、BF16、LoRA rank 8 / alpha 16 / dropout 0.05；目标模块为 `q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj`。
- 推理：基座模型加载后挂载 `checkpoints/adapter_formal`，可切换 Base 与 QLoRA。
- 原典：CBETA 八部经典，来源、版本和文件 SHA256 记录在 `data/raw/cbeta/sources.json`。

## 环境

已验证的远端环境为 Python 3.11.17、PyTorch 2.6.0+cu124、Transformers 4.57.6、PEFT 0.18.1、bitsandbytes 0.48.2、Accelerate 1.11.0、datasets 4.0.0、LLaMA-Factory 0.9.5。仓库不包含基座模型，运行前需将官方模型放到 `models/DeepSeek-V2-Lite-Chat`，并在兼容的 Linux CUDA 环境安装上述依赖。

以下命令均从项目根目录执行：

```bash
python src/test_model.py --max-new-tokens 96
python src/preprocess.py
python src/build_dataset.py
OMP_NUM_THREADS=8 python src/train.py --config configs/qlora_formal.yaml --check-data
OMP_NUM_THREADS=8 python src/train.py --config configs/qlora_formal.yaml
python src/test_model.py --adapter checkpoints/adapter_formal --max-new-tokens 128
```

训练脚本在目标 Adapter 目录非空时拒绝覆盖。正式训练共 48 条训练样本、8 条验证样本；整部经典留出，独立测试题不参与训练和模型选择。数据仅 64 条，属于小规模技术实验，未经佛学专家或人工独立审校，不满足 500–1500 条高质量问答的建议规模。

## 对话

```bash
python src/chat.py --mode qlora
python src/chat.py --mode base
```

交互中可输入 `/mode base`、`/mode qlora` 切换模型，`/clear` 清空历史，`/quit` 退出。可通过 `--temperature`、`--top-p`、`--max-new-tokens` 调整生成参数。

## 评测

```bash
python src/evaluate.py --questions data/evaluation/test_questions.json \
  --adapter checkpoints/adapter_formal \
  --output outputs/evaluation/base_vs_formal.json
python src/analyze_evaluation.py \
  --input outputs/evaluation/base_vs_formal.json \
  --output outputs/evaluation/lexical_summary.json
python src/plot_results.py
python src/verify_project.py
```

评测题覆盖知识理解、学习成长、工作事业、情绪压力、人际关系、日常生活、出处追问和不可确认问题。解码使用固定贪心设置。自动引文检查只判断带引号片段是否能在八部 CBETA 原文中逐字找到（忽略空白）；它不能判断引文上下文是否正确，也不能评价义理、事实准确性或回答质量。`analyze_evaluation.py` 的引用标记和拒答指标是词面代理量，不是人工评分。

## 当前结果与限制

正式 Adapter 位于 `checkpoints/adapter_formal`，已完成重新加载和中文推理检查；该次推理分配显存峰值为 11248.4 MiB（约 10.98 GiB）。正式配置对应 18 个预期优化步，当前可回读终端明确验证至第 15 步，Adapter 已保存；可取得的验证 loss 记录为 epoch 1 的 3.5424、epoch 2 的 3.2403。训练结束指标文件没有保留下来，因此完整 18 步审计、最终 epoch 的 loss、训练耗时和训练峰值显存不在当前可复核记录中。训练观察值及 Adapter 重载检查见 `outputs/metrics/observed_training_and_reload.json`；该文件是终端观察转录，不是完整 Trainer 日志。配对评测原始输出见 `outputs/evaluation/2024211891.json`，自动词面统计见 `outputs/evaluation/lexical_summary.json`，分析与限制见 `reports/experiment_report.md`。

训练样本由原典段落支撑，但问答由 AI 辅助编写，未经专家审校；测试集仅 52 题。即使自动指标有变化，也不能据此认定微调提升了佛学正确性。正式报告应结合人工双人评分和更大、经审校且无泄漏的数据集重新验证。
