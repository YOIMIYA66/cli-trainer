# CLI Trainer Skill — AI Studio 无代码大模型微调

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-AI%20Studio-orange.svg)](https://aistudio.baidu.com/)

> 一句话触发，全程引导，不写一行训练代码，在百度 AI Studio 云端完成大模型 SFT 微调。

---

## 这个 Skill 能做什么

帮你在百度 AI Studio 星河社区完成大模型 SFT 微调，全程无需 GPU、无需配环境、无需写训练代码。

**支持的模型：**
- **文心 ERNIE 系列**（ERNIE-4.5-0.3B / 21B-A3B 全系）：PaddleFormers 框架，Full 参数微调
- **主流开源模型**（Qwen2.5、Llama 3、MiniCPM、Baichuan2 等）：LlamaFactory 框架，支持 SFT/Full、SFT/LoRA

---

## 快速开始

**第一步：确认环境**

```bash
# 需要 Python 3.8+，获取 AI Studio Access Token 后运行：
export AISTUDIO_ACCESS_TOKEN="your_token_here"
python3 scripts/train.py --env-check
```

Token 获取地址：https://aistudio.baidu.com/account/accessToken

**第二步：在 AI 助手里说一句话**

```
帮我微调一个 Qwen 模型，用来回答客服问题
```

**第三步：Skill 自动引导你完成全流程**

```
选模型 → 准备数据 → 推荐超参 → 提交训练 → 监控进度 → 产物发布
```

---

## 安装

### 方式一：通过 Claude Code / Ducc 安装（推荐）

在对话框中输入：
```
安装 no-code-train skill
```

### 方式二：手动克隆

```bash
git clone https://github.com/your-username/cli-trainer.git ~/.claude/skills/no-code-train
```

---

## 触发方式

在任何支持 Skill 的 AI 编程助手对话框里，说包含以下关键词的句子即可自动触发：

**核心关键词：** 微调 / 训练模型 / SFT / fine-tune / finetune / 无代码训练 / ERNIE 微调

**业务化表述（同样有效）：**
问答对 / 让模型学会 / 定制领域助手 / 专属领域 AI / 调教模型 / 业务知识注入 / 领域适配 / 医疗问答 / 客服问答 / 行业术语

**示例：**
```
帮我用自己的问答对微调一个 ERNIE 模型
我想让模型学会我们公司的业务知识
帮我定制一个医疗问答领域的 AI 助手
用这批客服对话数据调教一下 Qwen
```

---

## 核心能力

| 能力 | 说明 |
|------|------|
| 白名单选型 | 列出平台当前可训练的模型，不用自己猜名称 |
| 数据格式检查 | 提交前验证 ERNIE/Alpaca/ShareGPT 格式，拦截格式错误 |
| 数据集上传校验 | 上传后核验 `is_lfs:false`，避免 `waiting_data` 静默卡死 |
| 超参自动推荐 | 根据数据量推荐 epochs/learning_rate/batch_size，逐行解释 |
| 实时轮询 | 持续打印阶段状态，自动打开 Tensorboard 看板 |
| 训练报告 | 输出 loss 曲线、下降幅度、ASCII 折线图、收敛健康诊断 |
| 模型仓库发布 | 训练完成后自动推送 README 模型卡片，设置多语言/任务/框架标签，确认公开状态 |
| LoRA 产物验证 | 确认合并导出完成，通过 API 调用验证产物可用 |

---

## 完整流程

```
 1. 环境自检   检查 Python、依赖、网络连通性和 Token 状态
 2. 鉴权       确认 AI Studio Access Token
 3. 选模型     展示白名单，选定基座模型
 4. 验数据     检查训练数据格式是否适配目标模型
 5. 准备数据集 用内置数据集，或上传自己的数据到 AI Studio 仓库
 6. 推荐超参   根据数据量推荐参数，等你确认后才提交
 7. 提交训练   一键提交，拿到任务 ID
 8. 实时监控   持续打印状态，进入 running 后自动打开 Tensorboard
 9. 训练报告   汇报 loss 趋势、健康状态和模型仓库地址
10. 模型发布   自动写 README 模型卡片，设置标签，确认公开
```

---

## 内置数据集

没有自己的数据？可以用这些内置数据集快速体验：

| 数据集 | 框架 | 大小 | 适合场景 |
|--------|------|------|---------|
| `lmtyyz/self-cognition` | LlamaFactory | 23 KB | 验证流程是否跑通（极小） |
| `lmtyyz/lima` | LlamaFactory | — | 高质量对话，入门首选 |
| `lmtyyz/alpaca-gpt4-data-zh` | LlamaFactory | 31 MB | 中文通用指令跟随 |
| `lmtyyz/school-math-0.25M` | LlamaFactory | 119 MB | 数学能力增强 |
| `lmtyyz/Multilingual-Thinking` | ERNIE | — | 多语言推理链 |

---

## 环境要求

- **Python 3.8+**（唯一必需的本地依赖）
- **AI Studio Access Token**：[获取地址](https://aistudio.baidu.com/account/accessToken)
- **网络**：需能访问 `train.aistudio-app.com`
- **可选：** web-access skill（自建数据集时自动创建仓库）
- **可选：** Playwright MCP（操作训练看板和模型元信息）

---

## 常见问题

**Q: `waiting_data` 超过 10 分钟？**
按顺序检查：① `repo_id` 是否来自详情页 ② `--train-file` 和仓库实际文件名是否一致 ③ 文件 `is_lfs` 是否为 `false`。Skill 会自动诊断并提示。

**Q: 报"非 ERNIE 格式"？**
数据用了 Alpaca 格式。ERNIE 需要 `{"src": ["问题"], "tgt": ["回答"]}` 格式，`src`/`tgt` 必须是列表。

**Q: 单次最大能训多大的模型？**
平台单卡约 32GB 显存，Full SFT 建议 7B 以下；7B+ 推荐用 LoRA。

**Q: 想迭代效果怎么办？**
平台只允许从官方基础模型开始训练。迭代时需合并新旧数据集后重新训练。

---

## 项目结构

```
cli-trainer/
├── SKILL.md              # Skill 主指令文件（AI 读取）
├── scripts/
│   └── train.py          # 所有平台交互逻辑（1600+ 行）
├── references/
│   ├── datasets.md       # 内置数据集列表
│   ├── aistudio_sdk_upload.md  # SDK 上传说明
│   └── model_whitelist.yaml    # 可训练模型白名单
└── tests/
    └── test_train.py     # 单元测试
```

---

## License

Apache License 2.0 — 详见 [LICENSE](LICENSE)
