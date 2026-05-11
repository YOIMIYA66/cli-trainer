# 内置推荐数据集

所有数据集已托管于 AiStudio 星河社区，公开可用，训练文件均为 `train.jsonl`。
当用户不知道用什么数据时，根据目标模型类型和场景从下方推荐。

---

## LlamaFactory 格式（开源模型 SFT/LoRA/DPO）

### Alpaca 格式（4 个）
`{"instruction": "...", "input": "...", "output": "..."}`

| 数据集路径 | 大小 | 描述 | 适合场景 |
|-----------|------|------|---------|
| `lmtyyz/alpaca-gpt4-data-zh` | 31 MB | 中文 Alpaca GPT-4 指令 | 通用中文指令跟随，入门首选 |
| `lmtyyz/alpaca-gpt4-data-en` | 38 MB | 英文 Alpaca GPT-4 指令 | 通用英文指令跟随 |
| `lmtyyz/school-math-0.25M` | 119 MB | 数学题 0.25M 条 | 增强模型数学计算能力 |
| `lmtyyz/lima` | 2.8 MB | LIMA 高质量对话 | 数据少但质量高，快速验证流程 |

### ShareGPT 格式（4 个）
`{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}`

| 数据集路径 | 大小 | 描述 | 适合场景 |
|-----------|------|------|---------|
| `lmtyyz/self-cognition` | 23 KB | 自我认知（我是谁）| 极小，仅验证训练流程是否跑通 |
| `lmtyyz/ShareGPT-Chinese-zh` | 181 MB | 中文多轮对话 | 增强中文多轮对话能力 |
| `lmtyyz/WizardLM-evol-instruct-V2` | 316 MB | 英文进化指令（WizardLM）| 全面提升英文指令理解和遵循 |
| `lmtyyz/Agent-FLAN` | 137 MB | Agent 工具调用微调数据 | 训练模型具备 Agent / 工具调用能力 |

**使用方式：**
```bash
python3 "$SKILL_PATH/scripts/train.py" --submit \
  --base-model "ModelHub/Qwen2.5-7B-Instruct" \
  --train-type "SFT/LoRA" \
  --train-data "lmtyyz/alpaca-gpt4-data-zh" \
  --train-file "train.jsonl" \
  --params '{"num_train_epochs": 3, "per_device_train_batch_size": 4, "learning_rate": 2e-4, "cutoff_len": 512, "lora_rank": 8, "lora_alpha": 16, "fp16": true}'
```

---

## PaddleFormers 格式（ERNIE 系列 SFT/Full）

`{"src": ["问题"], "tgt": ["回答"]}`

| 数据集路径 | 大小 | 描述 | 适合场景 |
|-----------|------|------|---------|
| `lmtyyz/Multilingual-Thinking` | 2 MB | 多语言推理链数据 | 小体量，快速上手验证流程 |
| `lmtyyz/Nemotron-SFT-Safety-v1` | 77 MB | 安全对齐 SFT 数据 | 增强安全拒绝和对齐能力 |
| `lmtyyz/Bespoke-Stratos-17s` | 288 MB | Bespoke-Stratos 推理数据 | 大体量，提升推理和思维链能力 |

**使用方式：**
```bash
python3 "$SKILL_PATH/scripts/train.py" --submit \
  --base-model "PaddlePaddle/ERNIE-4.5-0.3B-PT" \
  --train-type "SFT/Full" \
  --train-data "lmtyyz/Multilingual-Thinking" \
  --train-file "train.jsonl" \
  --params '{"num_train_epochs": 3, "per_device_train_batch_size": 4, "learning_rate": 5e-5, "max_seq_len": 512, "bf16": true}'
```

---

## 场景推荐速查

| 用户目标 | 推荐数据集 | 框架 |
|---------|-----------|------|
| 不知道从哪开始、先跑通流程 | `lmtyyz/self-cognition`（23 KB）| LlamaFactory |
| 通用中文能力 | `lmtyyz/alpaca-gpt4-data-zh` | LlamaFactory |
| 通用英文能力 | `lmtyyz/alpaca-gpt4-data-en` 或 `WizardLM-evol-instruct-V2` | LlamaFactory |
| 数学 / 推理 | `lmtyyz/school-math-0.25M` | LlamaFactory |
| 中文多轮对话 | `lmtyyz/ShareGPT-Chinese-zh` | LlamaFactory |
| Agent / 工具调用 | `lmtyyz/Agent-FLAN` | LlamaFactory |
| ERNIE 快速体验 | `lmtyyz/Multilingual-Thinking` | PaddleFormers |
| ERNIE 推理增强 | `lmtyyz/Bespoke-Stratos-17s` | PaddleFormers |

也可运行 `--list-datasets` 直接在命令行查看完整列表。
