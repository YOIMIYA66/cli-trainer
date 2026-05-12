---
name: no-code-train
description: 引导用户在百度 AI Studio 无代码训练平台完成大模型微调，全程无需写训练代码。支持文心 ERNIE 系列（SFT/Full）和开源模型（Qwen、LLaMA 等，SFT/LoRA）。当用户提到"微调"、"训练模型"、"无代码训练"、"SFT"、"LoRA"、"fine-tune"、"AiStudio 训练"、"ERNIE 微调"、"模型训练"、"finetune"时，优先使用这个 skill。
compatibility: 需要 Python 3 和 requests 包；用户需要提供 AI Studio Access Token
---

# 无代码训练 Skill

帮助用户在 AI Studio 云端完成大模型微调，不需要 GPU、不需要搭环境、不需要写训练代码。

用 `scripts/train.py` 执行所有平台交互，脚本路径通过 `$SKILL_PATH` 引用：

```bash
python3 "$SKILL_PATH/scripts/train.py" --check-data my_data.jsonl
```

---

## !! 绝对禁止事项（最高优先级）

1. **禁止自己猜测或硬写超参数然后提交。** 必须先运行 `--suggest-params`，把推荐值展示给用户，等用户确认再提交。
2. **禁止在用户明确确认前调用 `--submit`。** 展示参数时逐行解释含义，问"需要调整哪些？没问题就提交"，等用户回复。

---

## 执行顺序

0. **环境自检** — 首次使用时先跑一次，自动检测 Python 版本、依赖包、网络连通性，requests 缺失会自动安装
1. **鉴权** — 确认有 Access Token，没有就引导去 https://aistudio.baidu.com/account/accessToken 获取
2. **选模型** — 运行 `--list-models` 展示白名单，让用户选，不要让用户自己猜名称
3. **验数据** — 运行 `--check-data` 检查格式，提前发现错误
4. **上传数据并设为公开** — 数据集必须公开，训练后端才能拉取。用命令行上传时**默认加 `-p`**；若用网页上传，完成后立即进详情页 → 设置 → 设为公开。**这一步必须在提交前完成，否则提交时直接报权限错误。**
5. **推荐超参** — 运行 `--suggest-params`，展示结果，**等用户确认后才提交**
6. **提交** — 运行 `--submit`，拿到 jobId
7. **设 CronCreate** — 提交后立刻设定定时查状态，不让用户干等（见下方模板）
8. **取结果** — 训练完成后给出模型仓库地址，引导去 Playground 测效果

---

## 各步骤关键细节

### 环境自检

```bash
python3 "$SKILL_PATH/scripts/train.py" --env-check
```

检查内容：Python 版本（需 3.8+）、requests 包（缺失时自动 pip install）、网络连通性。
如果 python3 命令本身不存在，告知用户安装 Python 3.8+：macOS 用 `brew install python3`，其他平台参考 https://www.python.org/downloads/

### Token

三种方式（推荐环境变量）：
```bash
export AISTUDIO_ACCESS_TOKEN="your_token_here"           # 推荐
python3 "$SKILL_PATH/scripts/train.py" --verify-token    # 验证是否有效
```

### 选模型

```bash
python3 "$SKILL_PATH/scripts/train.py" --list-models
```

- 只能用白名单里的模型，HuggingFace/ModelScope 原始路径不可用
- 单卡最大 32B 以下，超过会 OOM
- **训练完的模型不能作为下一轮的 baseModel**（平台服务端白名单强制拦截）。想迭代效果只能：换数据集重新从官方基础模型训练，或联系平台申请把自定义模型加入白名单
- ERNIE → 框架 PaddleFormers，trainType 只能 `SFT/Full`，数据格式 src/tgt
- 开源模型 → 框架 LlamaFactory，trainType 用 `SFT/LoRA`，数据格式 Alpaca 或 ShareGPT
- Qwen3 系列（PaddleNLP/Qwen3-*）在平台上可能失败（Transformers 版本问题），推荐用 `ModelHub/Qwen2.5-*`

**ERNIE 变体选择指引：**

| 变体后缀 | 适合场景 |
|---------|---------|
| `-PT` | 想从预训练权重开始 SFT，效果通常更好（推荐）|
| `-Paddle` | 基础模型已有对话能力，在此基础上加领域知识 |
| `-Base-PT` | 需要纯 Base 模型做实验 |
| `-Base-Paddle` | Base + 指令微调的组合 |

### 数据格式

**ERNIE 格式（src/tgt 值必须是列表）：**
```jsonl
{"src": ["问题"], "tgt": ["回答"]}
```
最常见错误：用了 Alpaca 格式，任务会跑起来但 poller 阶段报"非 ERNIE 格式"。

**LlamaFactory 格式（Alpaca）：**
```jsonl
{"instruction": "问题", "input": "", "output": "回答"}
```

**LlamaFactory 格式（ShareGPT）：**
```jsonl
{"conversations": [{"from": "human", "value": "问题"}, {"from": "gpt", "value": "回答"}]}
```

```bash
python3 "$SKILL_PATH/scripts/train.py" --check-data 数据文件.jsonl
```

数据规模参考：50-500 条（验证流程）/ 1k-10k（场景微调）/ 10k+（全面提升）

### 上传数据

- 没有自己数据：先读取 `$SKILL_PATH/references/datasets.md`，根据用户目标推荐合适的数据集，再运行 `--list-datasets` 展示完整列表
- 自己上传，**一律使用以下命令，`-p` 是必须的**（不加则数据集私有，提交时直接报 `code=10004` 权限错误）：

```bash
pip install aistudio-sdk
aistudio config -t 你的Token
aistudio dataset create -n 数据集名称 -f 数据文件.jsonl -p
```

- 网页上传同理：创建完成后必须立即进详情页 → 设置 → 设为公开，再提交训练。
- **常见坑 1：** 建了仓库没上传文件，任务卡 `waiting_data` 不动
- **常见坑 2（已实测）：** 忘记加 `-p`，提交时平台直接返回 `code=10004 无数据集查看权限`，任务创建失败。

### 推荐超参并确认

```bash
python3 "$SKILL_PATH/scripts/train.py" --suggest-params 数据文件.jsonl --model-type ernie
```

脚本输出推荐值后，逐行解释给用户，明确问："需要调整哪些？没问题就提交。" 等回复再往下走。

**ERNIE（PaddleFormers）主要参数：**

| 参数 | 类型 | 推荐值 | 说明 |
|------|------|-------|------|
| `num_train_epochs` | float | 3 | 训练轮数，数据少可调大到 5 |
| `per_device_train_batch_size` | int | 4 | 每步样本数，OOM 就调小到 2 |
| `learning_rate` | float | 5e-5 | 学习率，新手一般不用改 |
| `max_seq_len` | int | 512 | 最大序列长度，超过截断 |
| `max_steps` | int | -1 | -1 表示由 epochs 控制；设正数则固定步数 |
| `warmup_steps` | int | 50 | 预热步数，约总步数的 5-10% |
| `logging_steps` | int | 5 | 每几步打一次日志 |
| `bf16` | bool | true | 混合精度，节省显存 |

**LlamaFactory（开源模型）主要参数：**

| 参数 | 类型 | 推荐值 | 说明 |
|------|------|-------|------|
| `num_train_epochs` | float | 3 | 训练轮数 |
| `per_device_train_batch_size` | int | 4 | 批大小 |
| `learning_rate` | float | 2e-4 | LoRA 学习率通常比 Full 大 |
| `cutoff_len` | int | 512 | 最大序列长度 |
| `lora_rank` | int | 8 | rank 越大能力越强但显存更多 |
| `lora_alpha` | int | 16 | 一般设为 lora_rank 的 2 倍 |
| `fp16` | bool | true | 混合精度 |

### 提交

```bash
python3 "$SKILL_PATH/scripts/train.py" --submit \
  --base-model "PaddlePaddle/ERNIE-4.5-0.3B-PT" \
  --train-type "SFT/Full" \
  --train-data "用户名/数据集名" \
  --train-file "文件名.jsonl" \
  --params '{"num_train_epochs": 3, "per_device_train_batch_size": 4, "learning_rate": 5e-5, "max_seq_len": 512, "bf16": true}'
```

注意：
- `--train-file` 必须指定，不传会报"未找到训练文件路径"
- 任务名 `--name` 不能含横杠，只能用字母/数字/下划线
- 每账号最多 30 个模型仓库，满了用 `--output-repo` 复用已有仓库
- 可视化参数（`report_to`/`visualdl`）由平台后端自动管理，用户传了反而会报"不支持的参数"错误，无需手动传

### 提交后立刻设 CronCreate（不能跳过）

提交成功拿到 jobId 后，立刻用 `CronCreate` 设定自动查状态：
- `waiting_data`/`pending` 阶段：每 **3 分钟**查一次
- `running` 阶段：每 **8 分钟**查一次

CronCreate prompt 模板（替换 JOB_ID）：
```
运行以下命令查询训练任务状态并告知用户：
python3 "$SKILL_PATH/scripts/train.py" --status JOB_ID

根据返回状态，按如下方式响应：

- waiting_data / pending：
  明确告知用户"当前状态 [状态]，平台正在下载模型/调度 GPU 资源，这是正常现象，
  可能需要 1-15 分钟，请继续等待。"
  不要只是沉默地检查，每次都要主动发一条消息告知用户进展。

- running（首次检测到，即刚从 pending 切过来）：
  1. 立即运行以下命令在浏览器打开可视化看板：
     python3 "$SKILL_PATH/scripts/train.py" --open-tb JOB_ID
  2. 告知用户训练已开始，并说明看板已自动在浏览器打开
  3. 将 CronCreate 间隔调整为 8 分钟（取消当前 cron，重新创建）
  注意：只在首次切到 running 时执行上述步骤（切换 cron 间隔是天然的"首次"标记）。

- succeeded：
  1. 告知用户训练已完成，给出模型仓库地址（通过 API 调用测试：model 字段填仓库路径）
  2. 立即运行训练汇报命令并把结果告知用户：
     python3 "$SKILL_PATH/scripts/train.py" --train-summary JOB_ID
     （训练完成后也可随时重新运行）
  3. 取消 cron job，停止轮询

- failed / cancelled：
  告知用户失败原因（如有），建议运行 --logs JOB_ID --system 查看详情。
  取消 cron job，停止轮询。
```

任务结束后主动取消 cron job。

### 监控进度

```bash
python3 "$SKILL_PATH/scripts/train.py" --status JOB_ID    # 查一次
python3 "$SKILL_PATH/scripts/train.py" --logs JOB_ID      # 查 stdout loss
```

状态含义：`waiting_data`（下载中，1-10min）→ `pending`（等 GPU，1-5min）→ `running`（训练中）→ `succeeded`

- `--logs` 查 stdout 是最可靠的 loss 监控方式，ERNIE/LlamaFactory 都支持
- `--train-summary` 会从日志解析 loss/lr 并输出训练趋势，训练完成后依然有效
- Tensorboard/VisualDL 仅训练中（running 阶段）有效，训练结束后数据流关闭，不再展示
- `--poll` 会阻塞终端，对话场景下用 CronCreate 替代

### 训练完成后

运行 `--train-summary` 获取 loss/lr 汇报，再用 `--eval-guide` 生成测试问题建议，通过 API 调用模型测效果：
```bash
python3 "$SKILL_PATH/scripts/train.py" --train-summary 训练任务ID
python3 "$SKILL_PATH/scripts/train.py" --eval-guide 训练数据.jsonl
```

先测训练集内的问题，再测训练集外同类问题，对比泛化能力。

---

## 常见错误速查

| 错误 | 原因 | 解决 |
|------|------|------|
| `code=10004 无数据集查看权限` | 数据集未设为公开 | `aistudio dataset create` 加 `-p`；或网页进详情页 → 设置 → 设为公开 |
| `waiting_data` 超过 10 分钟 | 数据集文件未上传 | 私有数据集在提交时就会报 `code=10004`；卡 waiting_data 是文件未上传，确认后重新提交 |
| "非 ERNIE 格式" | Alpaca 格式，或 src/tgt 值是字符串非列表 | `{"src": ["问题"], "tgt": ["回答"]}` |
| "类型错误：期望 float，实际 str" | 超参数是字符串 | 去掉引号：`3` 不是 `"3"` |
| "PaddleFormers 仅支持 SFT/Full" | ERNIE 传了 LoRA | `--train-type "SFT/Full"` |
| 任务名报错 | name 含横杠 | 改用下划线 |
| 模型库满 | 超过 30 个仓库 | 删旧仓库或 `--output-repo` 复用 |
| 微调后的模型再训练报错 | 平台白名单只允许官方模型 | 只能从官方基础模型重新训练，迭代时合并数据集 |
| code=401 | Token 过期 | 重新获取 Access Token |
| Qwen3 架构识别失败 | 平台 Transformers 版本旧 | 改用 `ModelHub/Qwen2.5-*` |
| "未找到训练文件路径" | 未传 `--train-file` | 加 `--train-file 文件名.jsonl` |

```bash
python3 "$SKILL_PATH/scripts/train.py" --cancel JOB_ID  # 取消卡住的任务
```

---

## 脚本参数速查

```
--env-check                     检查本地环境（Python 版本、依赖、网络）
--verify-token                  验证 Token
--list-models                   列出可用模型（白名单）
--list-datasets                 列出内置推荐数据集
--check-data <file>             检查数据格式
--suggest-params <file> [--model-type ernie|llama]  推荐超参数
--submit ...                    提交训练任务
--status <job_id>               查看任务状态
--logs <job_id> [--system]      查看训练日志（stdout loss）
--train-summary <job_id>         训练完成后汇报 loss/lr 趋势和健康状态
--poll <job_id>                 持续轮询（阻塞终端，对话场景不推荐）
--cancel <job_id>               取消任务
--eval-guide <file>             生成测试问题和判断标准

--api-key TOKEN / --env-file FILE / --base-url URL
```

---

## References 索引

| 文件 | 何时加载 |
|------|---------|
| `references/datasets.md` | 用户没有自己的数据，或问"用什么数据集好"时，先读此文件再推荐 |

