#!/usr/bin/env python3
"""AiStudio 无代码训练 CLI

用法：
  train.py --verify-token                         验证 Access Token
  train.py --check-data <file>                   检查数据格式
  train.py --suggest-params <file> [--model-type ernie|llama]  推荐超参
  train.py --submit --base-model M --train-type T --train-data D/N --train-file F [--params JSON]
  train.py --status <job_id>                     查看任务状态
  train.py --poll <job_id> [--interval N]        持续轮询（训练开始后自动打开 Tensorboard）
  train.py --open-tb <job_id>                    打开 Tensorboard 看板
  train.py --logs <job_id> [--system]            查看日志
  train.py --cancel <job_id>                     取消任务
  train.py --eval-guide <file>                   根据训练数据生成测试问题
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
import re
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:
    print("缺少依赖：requests\n安装命令：pip install requests")
    sys.exit(1)

BASE_URL = "https://train.aistudio-app.com"
TOKEN_ENV_VARS = ("AISTUDIO_ACCESS_TOKEN", "AISTUDIO_API_KEY")
WHITELIST_PATH = Path(__file__).parent.parent / "references" / "model_whitelist.yaml"

# --------------------------- whitelist ---------------------------- #

def _load_whitelist() -> dict[str, dict[str, str]]:
    """解析 model_whitelist.yaml，不依赖 PyYAML。"""
    if not WHITELIST_PATH.exists():
        return {}
    result: dict[str, dict[str, str]] = {}
    current: str | None = None
    for line in WHITELIST_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not line.startswith(" ") and stripped.endswith(":"):
            current = stripped[:-1]
            result[current] = {}
        elif current and ":" in line:
            k, _, v = line.partition(":")
            result[current][k.strip()] = v.strip().strip('"').strip("'")
    return result


def _whitelist_tool(model: str) -> str:
    """返回模型对应的 train_tool，不在白名单返回空字符串。"""
    return _load_whitelist().get(model, {}).get("train_tool", "")

# ---------------------------- auth ---------------------------- #

def load_token(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key.strip()

    if args.env_file:
        p = Path(args.env_file)
        if not p.exists():
            die(f"env 文件不存在：{p}")
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            k, _, v = line.partition("=")
            if k.strip() in TOKEN_ENV_VARS and v.strip():
                return v.strip().strip('"').strip("'")
        die(f"env 文件中未找到 AISTUDIO_ACCESS_TOKEN 或 AISTUDIO_API_KEY")

    for var in TOKEN_ENV_VARS:
        val = os.environ.get(var, "").strip()
        if val:
            return val

    die(
        "未找到 Access Token。请通过以下任一方式提供：\n"
        "  export AISTUDIO_ACCESS_TOKEN='your_token'\n"
        "  --api-key 'your_token'\n"
        "  --env-file .aistudio.env\n\n"
        "获取地址：https://aistudio.baidu.com/account/accessToken"
    )
    return ""  # unreachable; die() raises SystemExit


def headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def api(method: str, path: str, token: str, base_url: str, **kwargs) -> dict:
    url = base_url.rstrip("/") + path
    resp: requests.Response
    try:
        resp = requests.request(method, url, headers=headers(token), timeout=30, **kwargs)
    except requests.exceptions.ConnectionError:
        die(f"无法连接到 {base_url}，请检查网络或 --base-url 参数")
        return {}
    except requests.exceptions.Timeout:
        die("请求超时，请稍后重试")
        return {}

    if resp.status_code == 401:
        die("Token 认证失败（401）。请确认 Access Token 是否正确，或重新获取：\nhttps://aistudio.baidu.com/account/accessToken")
    if resp.status_code == 404:
        die(f"资源不存在（404）：{path}")

    body: dict
    try:
        body = resp.json()
    except Exception:
        die(f"API 返回了非 JSON 响应（HTTP {resp.status_code}）：\n{resp.text[:300]}")
        return {}

    if body.get("code") != 0:
        msg = body.get("msg", "未知错误")
        die(f"API 错误（code={body.get('code')}）：{msg}")

    return body.get("data") or {}


def die(msg: str) -> None:
    print(f"\n错误：{msg}\n", file=sys.stderr)
    sys.exit(1)


# -------------------------- list models --------------------------- #

def cmd_list_models(_args: argparse.Namespace) -> None:  # noqa: ARG001
    wl = _load_whitelist()
    if not wl:
        print("白名单文件不存在或为空。")
        print(f"预期路径：{WHITELIST_PATH}")
        return

    paddle: list[str] = []
    llama: list[str] = []
    for model, meta in wl.items():
        tool = meta.get("train_tool", "")
        if tool == "paddleformers":
            paddle.append(model)
        elif tool == "llamafactory":
            llama.append(model)

    sep = "-" * 60
    print(f"\n{sep}")
    print("  平台可用模型（星河社区白名单）")
    print(f"{sep}")
    print("""
  说明：只有以下模型才能提交训练，不支持 HuggingFace 或其他
  来源的模型路径。算力限制：单卡最大支持 32B 参数以下的模型。
""")

    print("  【文心 ERNIE】  trainType: SFT/Full  数据格式: src/tgt JSONL")
    variant_note = {
        "-PT":          "预训练权重（推荐用于微调）",
        "-Paddle":      "指令微调权重（已有对话能力）",
        "-Base-PT":     "Base 预训练权重",
        "-Base-Paddle": "Base 指令微调权重",
    }
    sorted_variants = sorted(variant_note.items(), key=lambda x: -len(x[0]))
    for m in paddle:
        note = next((v for k, v in sorted_variants if m.endswith(k)), "")
        print(f"    {m:<48} {note}")

    print(f"\n  【开源模型】  trainType: SFT/LoRA  数据格式: Alpaca / ShareGPT JSONL")
    for m in llama:
        family = m.split("/")[-1].split("-")[0] if "/" in m else m
        print(f"    {m:<48} {family}")

    print(f"\n{sep}")
    print("  使用示例：")
    print("    --base-model 'PaddlePaddle/ERNIE-4.5-0.3B-PT' --train-type 'SFT/Full'")
    print("    --base-model 'ModelHub/Qwen2.5-7B-Instruct'   --train-type 'SFT/LoRA'")
    print(f"{sep}\n")


# ------------------------ list datasets ----------------------- #

# 内置推荐数据集（全部托管在 AiStudio 星河，文件为 train.jsonl）
_BUILTIN_DATASETS = {
    "ernie": [
        {
            "path": "lmtyyz/Multilingual-Thinking",
            "file": "train.jsonl",
            "size": "2 MB",
            "desc": "多语言推理链数据",
            "note": "小体量，适合快速上手",
        },
        {
            "path": "lmtyyz/Nemotron-SFT-Safety-v1",
            "file": "train.jsonl",
            "size": "77 MB",
            "desc": "安全对齐 SFT 数据（Nemotron）",
            "note": "中等体量",
        },
        {
            "path": "lmtyyz/Bespoke-Stratos-17s",
            "file": "train.jsonl",
            "size": "288 MB",
            "desc": "Bespoke-Stratos 推理数据",
            "note": "大体量，训练时间较长",
        },
    ],
    "llama": [
        {
            "path": "lmtyyz/self-cognition",
            "file": "train.jsonl",
            "size": "23 KB",
            "desc": "自我认知（我是谁）",
            "note": "极小，仅验证流程用",
        },
        {
            "path": "lmtyyz/lima",
            "file": "train.jsonl",
            "size": "2.8 MB",
            "desc": "LIMA 高质量对话",
            "note": "小体量，质量高",
        },
        {
            "path": "lmtyyz/alpaca-gpt4-data-zh",
            "file": "train.jsonl",
            "size": "31 MB",
            "desc": "中文 Alpaca GPT-4 指令",
            "note": "中等体量，通用中文指令",
        },
        {
            "path": "lmtyyz/alpaca-gpt4-data-en",
            "file": "train.jsonl",
            "size": "38 MB",
            "desc": "英文 Alpaca GPT-4 指令",
            "note": "中等体量，通用英文指令",
        },
        {
            "path": "lmtyyz/school-math-0.25M",
            "file": "train.jsonl",
            "size": "119 MB",
            "desc": "数学题 0.25M 条",
            "note": "大体量，适合增强数学能力",
        },
        {
            "path": "lmtyyz/ShareGPT-Chinese-zh",
            "file": "train.jsonl",
            "size": "181 MB",
            "desc": "中文多轮对话（ShareGPT）",
            "note": "大体量，中文多轮",
        },
        {
            "path": "lmtyyz/Agent-FLAN",
            "file": "train.jsonl",
            "size": "137 MB",
            "desc": "Agent 工具调用微调数据",
            "note": "大体量，适合训练 Agent 能力",
        },
        {
            "path": "lmtyyz/WizardLM-evol-instruct-V2",
            "file": "train.jsonl",
            "size": "316 MB",
            "desc": "英文进化指令数据（WizardLM）",
            "note": "最大，全面提升英文指令能力",
        },
    ],
}


def cmd_list_datasets(_args: argparse.Namespace) -> None:
    sep = "-" * 62
    print(f"\n{sep}")
    print("  内置推荐数据集（已托管于 AiStudio，可直接用于训练）")
    print(f"{sep}\n")
    print("  说明：这些数据集已公开，无需自己准备数据，适合快速体验。")
    print("        使用时直接将 --train-data 和 --train-file 填入即可。\n")

    print("  【ERNIE 专用】  格式：src/tgt JSONL  trainType: SFT/Full")
    for ds in _BUILTIN_DATASETS["ernie"]:
        path_w = f"{ds['path']:<42}"
        print(f"    {path_w} {ds['size']:>8}  {ds['desc']}  [{ds['note']}]")

    print()
    print("  【开源模型专用】  格式：Alpaca / ShareGPT JSONL  trainType: SFT/LoRA")
    for ds in _BUILTIN_DATASETS["llama"]:
        path_w = f"{ds['path']:<42}"
        print(f"    {path_w} {ds['size']:>8}  {ds['desc']}  [{ds['note']}]")

    print(f"\n{sep}")
    print("  使用示例（ERNIE + lima 数据集）：")
    print("    python3 train.py --submit \\")
    print("      --base-model 'PaddlePaddle/ERNIE-4.5-0.3B-PT' \\")
    print("      --train-type 'SFT/Full' \\")
    print("      --train-data 'lmtyyz/Multilingual-Thinking' \\")
    print("      --train-file 'train.jsonl' \\")
    print("      --params '{\"num_train_epochs\": 1, \"max_steps\": 200}'")
    print()
    print("  使用示例（Qwen2.5 + lima 数据集）：")
    print("    python3 train.py --submit \\")
    print("      --base-model 'ModelHub/Qwen2.5-3B-Instruct' \\")
    print("      --train-type 'SFT/LoRA' \\")
    print("      --train-data 'lmtyyz/lima' \\")
    print("      --train-file 'train.jsonl' \\")
    print("      --params '{\"num_train_epochs\": 1, \"lora_rank\": 8}'")
    print(f"{sep}\n")


# -------------------------- verify token ---------------------- #

def cmd_verify_token(args: argparse.Namespace) -> None:
    token = load_token(args)
    print("正在验证 Access Token ...")
    # 用一个无副作用的 GET 端点验证（非 200→401 表示 token 无效）
    url = args.base_url.rstrip("/") + "/v1/train/jobs/__verify_probe__/progress"
    resp: requests.Response
    try:
        resp = requests.get(url, headers=headers(token), timeout=10)
    except Exception as e:
        die(f"连接失败：{e}")
        return

    if resp.status_code == 401:
        print("Token 无效或已过期。")
        print("请重新获取：https://aistudio.baidu.com/account/accessToken")
        sys.exit(1)

    print("Token 有效！")


# -------------------------- check data ------------------------ #

def detect_format(line: str) -> str:
    """返回 'ernie' | 'alpaca' | 'sharegpt' | 'unknown'"""
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return "invalid_json"
    if isinstance(obj, dict):
        if "src" in obj and "tgt" in obj:
            return "ernie"
        if "instruction" in obj or ("input" in obj and "output" in obj):
            return "alpaca"
        if "conversations" in obj and isinstance(obj["conversations"], list):
            return "sharegpt"
    return "unknown"


def cmd_check_data(args: argparse.Namespace) -> None:
    path = Path(args.check_data)
    if not path.exists():
        die(f"文件不存在：{path}")

    print(f"正在检查：{path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    total = len(lines)

    # 过滤空行
    non_empty = [(i + 1, l) for i, l in enumerate(lines) if l.strip()]

    if not non_empty:
        die("文件为空，没有可用数据。")

    # 检测格式
    format_counts: dict[str, int] = {}
    errors: list[tuple[int, str, str]] = []

    for lineno, raw in non_empty:
        fmt = detect_format(raw)
        format_counts[fmt] = format_counts.get(fmt, 0) + 1
        if fmt in ("invalid_json", "unknown"):
            reason = "JSON 解析失败" if fmt == "invalid_json" else "字段不符合 ERNIE 或 Alpaca 格式"
            errors.append((lineno, raw[:80], reason))
        elif fmt == "ernie":
            # 额外校验：src/tgt 必须是 list，不能是裸字符串
            obj = json.loads(raw)  # 已在 detect_format 中解析成功，不会抛异常
            if not isinstance(obj.get("src"), list) or not isinstance(obj.get("tgt"), list):
                errors.append((lineno, raw[:80], 'ERNIE 格式错误：src/tgt 必须是列表 ["..."]，不能是裸字符串'))

    detected = max(format_counts.keys(), key=lambda k: format_counts[k])

    print(f"\n{chr(45) * 50}")
    print(f"  总行数（含空行）：{total}")
    print(f"  非空行数：        {len(non_empty)}")
    print(f"  检测到格式：      {_fmt_label(detected)}")

    if errors:
        print(f"\n  发现 {len(errors)} 处格式错误：")
        for lineno, preview, reason in errors[:10]:
            print(f"    第 {lineno} 行 [{reason}]：{preview}")
        if len(errors) > 10:
            print(f"    ... 还有 {len(errors) - 10} 处错误（只显示前 10 条）")
    else:
        print(f"  格式检查：        全部通过 [OK]")

    # 混合格式警告
    llama_fmts = {"alpaca", "sharegpt"}
    if "ernie" in format_counts and (format_counts.keys() & llama_fmts):
        print("\n  警告：文件中同时包含 ERNIE 格式和 Alpaca/ShareGPT 格式，可能混淆了两种数据集！")
    elif len(format_counts.keys() & llama_fmts) > 1:
        print("\n  警告：文件中同时包含 Alpaca 格式和 ShareGPT 格式，建议统一为一种格式。")

    # 如果 ERNIE 用户用了 Alpaca/ShareGPT 格式，提示转换
    if detected == "alpaca":
        print("""
  提示：如果要训练 ERNIE 模型，需要将格式转换为 ERNIE 格式：
    {"src": ["问题或指令"], "tgt": ["期望回答"]}

  快速转换脚本（Alpaca → ERNIE）：
    import json
    with open("input.jsonl") as fin, open("output.jsonl", "w") as fout:
        for line in fin:
            d = json.loads(line)
            text = d.get("instruction", "")
            if d.get("input"):
                text += "\\n" + d["input"]
            fout.write(json.dumps({"src": [text], "tgt": [d["output"]]}, ensure_ascii=False) + "\\n")
""")
    elif detected == "sharegpt":
        print("""
  提示：如果要训练 ERNIE 模型，需要将格式转换为 ERNIE 格式：
    {"src": ["问题或指令"], "tgt": ["期望回答"]}

  快速转换脚本（ShareGPT → ERNIE）：
    import json
    with open("input.jsonl") as fin, open("output.jsonl", "w") as fout:
        for line in fin:
            d = json.loads(line)
            convs = d.get("conversations", [])
            for i in range(0, len(convs) - 1, 2):
                human = convs[i].get("value", "") if convs[i].get("from") == "human" else ""
                gpt   = convs[i+1].get("value", "") if convs[i+1].get("from") == "gpt" else ""
                if human and gpt:
                    fout.write(json.dumps({"src": [human], "tgt": [gpt]}, ensure_ascii=False) + "\\n")
""")

    # 样本量建议
    n = len(non_empty) - len(errors)
    print(f"\n  有效样本数：{n}")
    if n < 50:
        print("  建议：样本量偏少（< 50），仅适合验证流程是否跑通，效果不保证。")
    elif n < 1000:
        print("  建议：样本量适合验证特定场景，期望全面提升建议达到 1,000 条以上。")
    elif n < 10000:
        print("  建议：样本量良好，适合特定场景微调。")
    else:
        print("  建议：样本量充足，适合较全面的能力提升训练。")

    print(f"{chr(45) * 50}\n")

    if errors:
        print("存在格式错误，建议修复后再提交训练任务。")
        sys.exit(1)
    elif detected == "alpaca":
        print("Alpaca 格式检查通过。")
        print("-> 训练开源模型（Qwen/DeepSeek 等）：可以直接提交。")
        print("-> 训练 ERNIE 模型：需要先用上面的脚本转换为 src/tgt 格式。")
    elif detected == "sharegpt":
        print("ShareGPT 格式检查通过。")
        print("-> 训练开源模型（Qwen/DeepSeek 等）：可以直接提交。")
        print("-> 训练 ERNIE 模型：需要先转换为 src/tgt 格式。")
    else:
        print("ERNIE 格式检查通过，可以提交训练任务。")


def _fmt_label(fmt: str) -> str:
    return {
        "ernie": "ERNIE 格式（src/tgt）",
        "alpaca": "Alpaca 格式（instruction/input/output）",
        "sharegpt": "ShareGPT 格式（conversations）",
        "unknown": "未知格式",
        "invalid_json": "JSON 解析失败",
    }.get(fmt, fmt)


# ------------------------ suggest params --------------------- #

def cmd_suggest_params(args: argparse.Namespace) -> None:
    path = Path(args.suggest_params)
    if not path.exists():
        die(f"文件不存在：{path}")

    lines = [l for l in path.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    # 只计有效 JSON 行（与 eval-guide 保持一致）
    valid_lines = []
    for l in lines:
        try:
            json.loads(l)
            valid_lines.append(l)
        except Exception:
            pass
    n = len(valid_lines)
    if not n:
        die("文件中没有可解析的有效 JSON 行，请先用 --check-data 检查数据格式。")

    # 估算平均序列长度（用有效行）
    sample = valid_lines[:min(100, n)]
    avg_len = sum(len(l) for l in sample) / len(sample) if sample else 200
    suggested_seq_len = min(2048, max(128, int(math.ceil(avg_len / 64) * 64)))

    model_type = (args.model_type or "ernie").lower()

    print(f"\n{chr(45) * 50}")
    print(f"  数据集：{path.name}")
    print(f"  样本数：{n}")
    print(f"  平均字符长度：{avg_len:.0f}  ->  建议 max_seq_len / cutoff_len：{suggested_seq_len}")
    print(f"  模型类型：{model_type}")
    print("-" * 50)

    if model_type == "ernie":
        _suggest_ernie(n, suggested_seq_len)
    else:
        _suggest_llama(n, suggested_seq_len)


def _suggest_ernie(n: int, seq_len: int) -> None:
    if n < 500:
        epochs, batch, lr = 5, 4, 5e-5
        note = "数据量较少，适当增加 epochs"
    elif n < 5000:
        epochs, batch, lr = 3, 4, 5e-5
        note = "适中数据量，使用默认配置"
    else:
        epochs, batch, lr = 3, 8, 3e-5
        note = "数据量较大，可适当增大 batch"

    params = {
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch,
        "learning_rate": lr,
        "max_seq_len": seq_len,
        "warmup_steps": min(100, max(10, n // 10)),
        "logging_steps": 5,
        "save_steps": 500,
        "bf16": True,
    }
    _print_params(params, note, "PaddleFormers（ERNIE）")


def _suggest_llama(n: int, seq_len: int) -> None:
    if n < 500:
        epochs, batch, lr = 5, 4, 2e-4
        rank = 8
        note = "数据量较少，适当增加 epochs"
    elif n < 5000:
        epochs, batch, lr = 3, 4, 2e-4
        rank = 8
        note = "适中数据量，使用默认 LoRA 配置"
    else:
        epochs, batch, lr = 3, 8, 1e-4
        rank = 16
        note = "数据量较大，可适当增大 lora_rank 和 batch"

    params = {
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch,
        "learning_rate": lr,
        "cutoff_len": seq_len,
        "lora_rank": rank,
        "lora_alpha": rank * 2,
        "lora_dropout": 0.05,
        "warmup_ratio": 0.03,
        "logging_steps": 5,
        "save_steps": 500,
        "fp16": True,
    }
    _print_params(params, note, "LlamaFactory（开源模型）")


def _print_params(params: dict, note: str, framework: str) -> None:
    print(f"\n  框架：{framework}")
    print(f"  说明：{note}")
    print(f"\n  推荐超参数：")
    for k, v in params.items():
        display = json.dumps(v) if isinstance(v, bool) else v
        print(f"    {k}: {display}")
    print(f"\n  JSON 格式（可直接粘贴到 --params）：")
    print(f"  {json.dumps(params, ensure_ascii=False)}")
    print(f"\n  提示：可以调整这些参数，调整后加 --params '{{...}}' 传给 --submit 命令。")
    print(f"  重要：超参数类型必须是数字/布尔，不能加引号。")
    print()


# ------------------------------ submit ----------------------- #

def cmd_submit(args: argparse.Namespace) -> None:
    token = load_token(args)

    # 数据集公开提醒（私有数据集提交时直接报 code=10004）
    print("⚠️  提交前请确认：数据集仓库必须设为【公开】，否则提交时会报权限错误。")
    print(f"   CLI 上传示例：aistudio dataset create -n 数据集名 -f 文件.jsonl -p  （-p 不能省）")
    print()

    # 白名单校验
    tool = _whitelist_tool(args.base_model)
    if not tool:
        wl = _load_whitelist()
        if wl:  # 白名单存在但模型不在里面
            die(
                f"模型 '{args.base_model}' 不在平台白名单中，无法提交训练。\n"
                f"请运行以下命令查看所有可用模型：\n"
                f"  python3 {Path(__file__).name} --list-models"
            )
    # 框架与 trainType 一致性检查
    VALID_TRAIN_TYPES = {"SFT/Full", "SFT/LoRA", "Post-PreTrain"}
    if args.train_type not in VALID_TRAIN_TYPES:
        # 尝试大小写纠正
        normalized = next((t for t in VALID_TRAIN_TYPES if t.lower() == args.train_type.lower()), None)
        hint = f"\n你可能想用：--train-type '{normalized}'" if normalized else ""
        die(
            f"trainType '{args.train_type}' 不合法，支持的类型：{', '.join(sorted(VALID_TRAIN_TYPES))}"
            + hint
        )
    if tool == "paddleformers" and args.train_type != "SFT/Full":
        die(
            f"文心 ERNIE 模型只支持 SFT/Full，不支持 {args.train_type}。\n"
            f"请改为：--train-type 'SFT/Full'"
        )
    if tool == "llamafactory" and args.train_type == "SFT/Full":
        die(
            f"开源模型使用 LlamaFactory，trainType 应为 SFT/LoRA，\n"
            f"不支持 SFT/Full。请改为：--train-type 'SFT/LoRA'"
        )

    # 构建请求体
    payload: dict[str, Any] = {
        "baseModel": args.base_model,
        "trainType": args.train_type,
        "trainData": [args.train_data],
    }

    if args.train_file:
        payload["trainDataFiles"] = {args.train_data: {"train": args.train_file}}

    if args.name:
        _validate_name(args.name)
        payload["name"] = args.name

    if args.description:
        payload["description"] = args.description

    if args.output_repo:
        payload["modelOutputRepo"] = args.output_repo

    if args.max_run_time:
        payload["maxRunTime"] = args.max_run_time

    hp: dict = {}
    if args.params:
        try:
            hp = json.loads(args.params)
        except json.JSONDecodeError as e:
            die(f"--params 不是合法的 JSON：{e}")
            return

    # 可视化参数（report_to/visualdl）由平台后端自动注入，用户无需传也不能传

    if hp:
        _validate_hyperparams(hp, args.train_type)
        payload["hyperparameters"] = hp

    print(f"\n提交训练任务：")
    print(f"  基底模型：{args.base_model}")
    print(f"  训练类型：{args.train_type}")
    print(f"  数据集：  {args.train_data}/{args.train_file or '（自动选择）'}")
    if payload.get("hyperparameters"):
        print(f"  超参数：  {json.dumps(payload['hyperparameters'])}")
    print()

    data = api("POST", "/v1/train/jobs", token, args.base_url, json=payload)
    job_id = data.get("jobId", "")

    if not job_id:
        die(f"提交成功但未返回 jobId，响应：{data}")

    print(f"任务已提交！\n")
    print(f"  Job ID：{job_id}")
    print(f"\n查看状态：")
    print(f"  python3 {Path(__file__).name} --status {job_id}")
    print(f"\n持续轮询进度：")
    print(f"  python3 {Path(__file__).name} --poll {job_id}")


def _validate_name(name: str) -> None:
    import re
    if not re.match(r'^[a-zA-Z0-9_一-鿿]*$', name):
        die(f"任务名称不能包含横杠或特殊字符，只允许字母、数字、下划线。\n当前：{name}")


_STRING_PARAMS = {"report_to", "save_strategy", "lr_scheduler_type", "optim", "fsdp", "logging_dir", "output_dir"}

# 各框架平台不支持的参数（实测结论）
_LLAMAFACTORY_UNSUPPORTED = {"max_steps", "max_seq_len", "bf16", "warmup_steps"}
_PADDLEFORMERS_UNSUPPORTED = {"cutoff_len", "lora_rank", "lora_alpha", "lora_dropout", "warmup_ratio", "fp16"}


def _validate_hyperparams(hp: dict, train_type: str) -> None:
    errors = []
    for k, v in hp.items():
        if isinstance(v, str) and k not in _STRING_PARAMS:
            errors.append(f"  {k}: \"{v}\"  <- 应该是数字或布尔，不能是字符串")
    if errors:
        die("超参数类型错误，请去掉值的引号：\n" + "\n".join(errors))

    is_llama = train_type == "SFT/LoRA"
    is_paddle = train_type == "SFT/Full"

    if is_llama:
        bad = [k for k in hp if k in _LLAMAFACTORY_UNSUPPORTED]
        if bad:
            die(f"以下参数 LlamaFactory 不支持，提交会被平台拒绝：{bad}\n"
                f"  max_steps -> 改用 num_train_epochs\n"
                f"  max_seq_len -> 改用 cutoff_len\n"
                f"  bf16/warmup_steps -> 改用 fp16/warmup_ratio")
    elif is_paddle:
        bad = [k for k in hp if k in _PADDLEFORMERS_UNSUPPORTED]
        if bad:
            die(f"以下参数 PaddleFormers（ERNIE）不支持，提交会被平台拒绝：{bad}\n"
                f"  cutoff_len -> 改用 max_seq_len\n"
                f"  fp16/warmup_ratio -> 改用 bf16/warmup_steps\n"
                f"  lora_rank/lora_alpha -> ERNIE 不支持 LoRA")


# ------------------------------ env check -------------------- #

def cmd_env_check() -> None:
    import subprocess
    ok = True

    # Python 版本
    v = sys.version_info
    if v >= (3, 8):
        print(f"[OK] Python {v.major}.{v.minor}.{v.micro}")
    else:
        print(f"[X]  Python {v.major}.{v.minor}.{v.micro}  需要 3.8+")
        ok = False

    # requests
    try:
        import requests as _r
        print(f"[OK] requests {_r.__version__}")
    except ImportError:
        print("[X]  requests 未安装，正在尝试自动安装...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "requests", "-q"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("[OK] requests 安装成功")
        else:
            print(f"[X]  requests 安装失败，请手动运行：pip install requests\n{result.stderr.strip()}")
            ok = False

    # 网络连通性（只做 DNS，不发 API 请求）
    import socket
    try:
        socket.setdefaulttimeout(5)
        socket.getaddrinfo("train.aistudio-app.com", 443)
        print("[OK] 网络可访问 train.aistudio-app.com")
    except OSError:
        print("[X]  无法访问 train.aistudio-app.com，请检查网络或代理设置")
        ok = False

    print()
    if ok:
        print("环境自检通过，可以开始使用。")
    else:
        print("存在问题，请按上方提示修复后重试。")
        sys.exit(1)


# ------------------------------ status ----------------------- #

def cmd_status(args: argparse.Namespace) -> None:
    token = load_token(args)
    data = api("GET", f"/v1/train/jobs/{args.status}", token, args.base_url)
    _print_status(data)


def _print_status(data: dict) -> None:
    state = data.get("state", "未知")
    phase = data.get("currentPhase", "")
    error = data.get("errorMsg", "")
    tb_url = data.get("tensorboardUrl", "")
    log_url = data.get("logUrl", "")
    start = data.get("startTime", "")
    end = data.get("endTime", "")
    progress = data.get("trainingProgress") or {}
    output = data.get("modelOutputRepo")

    print(f"\n{chr(45) * 55}")
    print(f"  状态：        {_state_label(state, phase)}")
    if start:
        print(f"  开始时间：    {start}")
    if end:
        print(f"  结束时间：    {end}")

    if state == "running" and phase == "training" and progress:
        cur = progress.get("currentStep")
        tot = progress.get("totalSteps")
        elapsed = progress.get("elapsedTime", "")
        remaining = progress.get("remainingTime", "")
        loss = progress.get("trainLoss")

        if cur is not None and tot:
            pct = int(cur / tot * 100)
            bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
            print(f"  进度：        [{bar}] {pct}%  ({cur}/{tot} 步)")
        if elapsed:
            print(f"  已耗时：      {elapsed}")
        if remaining:
            print(f"  预计剩余：    {remaining}")
        if loss is not None:
            print(f"  训练 Loss：   {loss:.4f}")

    if state == "succeeded" and output:
        repo = output.get("modelRepo", "")
        branch = output.get("branch", "")
        print(f"\n  模型仓库：    {repo}")
        print(f"  分支：        {branch}")

    if error:
        print(f"\n  错误信息：    {error}")

    if tb_url:
        if state == "running":
            print(f"\n  Tensorboard（实时 Loss 曲线）：")
        else:
            print(f"\n  Tensorboard（训练已结束，实时数据流关闭，历史快照可能仍可访问）：")
        print(f"  {tb_url}")

    if log_url:
        print(f"\n  日志地址：    {log_url}")

    print(f"{chr(45) * 55}\n")

    if state == "waiting_data":
        print("提示：正在下载模型和数据集，通常需要 1-10 分钟。")
        print("如果超过 10 分钟，请检查数据集仓库是否已上传文件。")
    elif state == "pending":
        print("提示：任务已进入队列，等待 GPU 调度，通常 1-5 分钟。")
    elif state == "running" and phase == "training":
        if tb_url:
            print(f"Tensorboard 已自动打开（或手动访问）：{tb_url}")
        print("Loss 解读：持续下降 = 正常；每个 epoch 开始时突然上升 = 正常；")
        print("         一直不降或乱跳 = 数据质量有问题，先检查格式。")
    elif state == "succeeded":
        _print_test_guide(output or {})
    elif state == "failed":
        print("任务失败。常见原因：")
        print("  1. 数据格式错误（用 --check-data 检查）")
        print("  2. 超参数类型错误（值不能是字符串）")
        print("  3. 数据集为空（确认文件已上传）")


def _print_test_guide(output: dict) -> None:
    repo = output.get("modelRepo", "")
    separator = "=" * 55

    print(separator)
    print("  训练完成！以下是测试和使用模型的完整指南")
    print(separator)

    if repo:
        print(f"""
【第一步：API 调用测试模型（推荐先做）】

  训练完的模型通过 AiStudio API 调用，示例：

  curl -X POST https://aistudio.baidu.com/llm/lmapi/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -H "Authorization: token <your_token>" \\
    -d '{{
      "model": "{repo}",
      "messages": [{{"role": "user", "content": "你的测试问题"}}]
    }}'
""")

    print("""【第二步：验证微调效果的问题类型】

  推荐用以下几类问题测试，判断模型有没有真的学进去：

  ① 训练集内的问题（验证是否学到）
     -> 发一条 src 字段的原文，看回答是否接近 tgt
     -> 如果完全一样，可能过拟合；如果语义相近，效果良好

  ② 训练集外但同类型问题（验证是否泛化）
     -> 换个说法问同一类问题，看风格、术语是否统一
     -> 这是最重要的测试，体现微调是否有迁移效果

  ③ 无关问题（验证是否破坏基础能力）
     -> 问一些通用常识，确认模型没有遗忘基础能力
""")

    print("""【第三步：效果诊断标准】

  效果好的信号：
  [OK] 回答风格、用词与训练数据一致
  [OK] 对领域专有名词的处理明显改善
  [OK] 拒绝回答领域外问题（如果训练数据暗示这样做）

  效果不好的原因（按优先级排查）：
  1. 数据质量 — 答案是否准确、表述是否一致（最重要）
  2. 数据量 — 太少容易过拟合，建议至少 1,000 条
  3. 超参数 — 尝试调小学习率（1e-5）或增加 epochs
""")

    if repo:
        print(f"""【第四步：API 调用（用代码接入）】

  训练完的模型可以通过 AiStudio API 调用：
  模型仓库：{repo}
  参考文档：https://aistudio.baidu.com/doc/model-api
""")

    print("如果效果不达预期，告诉我具体现象，我来帮你分析原因。")
    print(separator)


def _open_url(url: str) -> None:
    try:
        webbrowser.open(url)
        print(f"已在浏览器打开：{url}")
    except Exception:
        print(f"请手动打开：{url}")


def _state_label(state: str, phase: str) -> str:
    labels = {
        "waiting_data": "等待数据 (waiting_data) — 正在下载模型/数据集",
        "pending": "等待调度 (pending) — 等待 GPU 分配",
        "running": f"训练中 (running/{phase})" if phase else "运行中 (running)",
        "succeeded": "训练成功 (succeeded) [OK]",
        "failed": "训练失败 (failed) [FAIL]",
        "cancelled": "已取消 (cancelled)",
    }
    return labels.get(state, state)


# -------------------------------- poll ------------------------ #

def cmd_poll(args: argparse.Namespace) -> None:
    token = load_token(args)
    interval = args.interval or 30
    job_id = args.poll

    print(f"开始轮询任务 {job_id}，每 {interval} 秒刷新一次（Ctrl+C 停止）\n")

    terminal_states = {"succeeded", "failed", "cancelled"}
    tb_opened = False

    try:
        while True:
            data = api("GET", f"/v1/train/jobs/{job_id}", token, args.base_url)
            state = data.get("state", "")
            phase = data.get("currentPhase", "")

            # 训练开始时自动打开 Tensorboard（只打开一次）
            if state == "running" and phase == "training" and not tb_opened:
                tb_url = data.get("tensorboardUrl", "")
                if tb_url:
                    print(f"\n  训练已开始，自动打开 Tensorboard 看板…")
                    _open_url(tb_url)
                    print()
                tb_opened = True

            # 单行进度
            progress = data.get("trainingProgress") or {}
            cur = progress.get("currentStep")
            tot = progress.get("totalSteps")
            remaining = progress.get("remainingTime", "")

            ts = time.strftime("%H:%M:%S")
            if cur and tot:
                pct = int(cur / tot * 100)
                bar = "#" * (pct // 10) + "-" * (10 - pct // 10)
                suffix = f"  [{bar}] {pct}%  剩余 {remaining}" if remaining else f"  [{bar}] {pct}%"
            else:
                suffix = ""

            print(f"  [{ts}]  {_state_label(state, phase)}{suffix}")

            if state in terminal_states:
                print()
                _print_status(data)
                break

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n轮询已停止。")
        print(f"再次查询：python3 {Path(__file__).name} --status {job_id}")


# -------------------------------- logs ------------------------ #

def cmd_logs(args: argparse.Namespace) -> None:
    token = load_token(args)
    job_id = args.logs
    log_file = "system.log" if args.system else "master/output.log"
    url = args.base_url.rstrip("/") + f"/v1/train/jobs/{job_id}/{log_file}"

    resp: requests.Response
    try:
        resp = requests.get(url, headers={**headers(token), "Range": "bytes=0-409500"}, timeout=30)
    except Exception as e:
        die(f"获取日志失败：{e}")
        return

    if resp.status_code == 404:
        die("日志文件不存在（任务可能还未开始，或 job_id 错误）")
    if resp.status_code == 401:
        die("Token 认证失败")

    content = resp.content.decode("utf-8", errors="replace")

    # 检测是否是 JSON 错误响应（如 {"code":10002,"msg":"任务不存在"}）
    if content.strip().startswith("{"):
        try:
            err = json.loads(content)
            if isinstance(err, dict) and err.get("code", 0) != 0:
                die(f"API 错误（code={err['code']}）：{err.get('msg', content)}")
        except json.JSONDecodeError:
            pass

    if not content.strip():
        print("日志为空（任务可能还在等待中）")
        return

    # 显示最后 100 行
    lines = content.splitlines()
    if len(lines) > 100:
        print(f"（共 {len(lines)} 行，显示最后 100 行）\n")
        lines = lines[-100:]

    print("\n".join(lines))


# -------------------------- train summary -------------------- #

def _fetch_log_content(job_id: str, token: str, base_url: str) -> str:
    url = base_url.rstrip("/") + f"/v1/train/jobs/{job_id}/master/output.log"
    try:
        resp = requests.get(url, headers={**headers(token), "Range": "bytes=0-819200"}, timeout=30)
    except Exception as e:
        die(f"获取训练日志失败：{e}")
        return ""
    if resp.status_code == 404:
        die("日志文件不存在（任务可能还未开始，或 job_id 错误）")
    if resp.status_code == 401:
        die("Token 认证失败")
    content = resp.content.decode("utf-8", errors="replace")
    if content.strip().startswith("{"):
        try:
            err = json.loads(content)
            if isinstance(err, dict) and err.get("code", 0) != 0:
                die(f"API 错误（code={err['code']}）：{err.get('msg', content)}")
        except json.JSONDecodeError:
            pass
    return content


def _parse_metrics(lines: list[str]) -> list[dict]:
    metrics: list[dict] = []
    for line in lines:
        # LlamaFactory: {'loss': 1.23, 'learning_rate': 2e-05, 'epoch': 0.5, ...}
        if "'loss'" in line or '"loss"' in line:
            try:
                m = re.search(r"\{[^}]+\}", line)
                if m:
                    d = ast.literal_eval(m.group(0))
                    if isinstance(d, dict) and "loss" in d:
                        metrics.append(d)
                        continue
            except Exception:
                pass
        # PaddleFormers: "loss: 1.234" or "loss=1.234"
        m = re.search(r"\bloss[:\s=]+([0-9]+\.[0-9]+(?:e[+-]?[0-9]+)?)", line, re.IGNORECASE)
        if m:
            entry: dict = {"loss": float(m.group(1))}
            lr_m = re.search(r"\blr[:\s=]+([0-9]+\.[0-9]+e[+-]?[0-9]+|[0-9]+\.[0-9]+)", line, re.IGNORECASE)
            if lr_m:
                try:
                    entry["learning_rate"] = float(lr_m.group(1))
                except Exception:
                    pass
            step_m = re.search(r"\bstep[:\s]+(\d+)", line, re.IGNORECASE)
            if step_m:
                entry["step"] = int(step_m.group(1))
            metrics.append(entry)
    return metrics


def _print_ascii_loss_chart(losses: list[float], width: int = 50, height: int = 10) -> None:
    """在终端打印 ASCII loss 折线图。"""
    if len(losses) < 2:
        return
    # 降采样到 width 个点
    step = max(1, len(losses) // width)
    sampled = losses[::step][:width]
    mn, mx = min(sampled), max(sampled)
    if mx == mn:
        return  # 无变化，不画

    print(f"\n[loss 折线图]  {mn:.4f} (低) ~ {mx:.4f} (高)")
    print(f"  {'loss':^{width}}")

    for row in range(height, -1, -1):
        threshold = mn + (mx - mn) * row / height
        line = ""
        for v in sampled:
            line += "*" if v >= threshold - (mx - mn) / height / 2 else " "
        label = f"{threshold:.4f}" if row % (height // 2 or 1) == 0 else "      "
        print(f"  {label} |{line}")

    print(f"  {'':6} +{'-' * len(sampled)}")
    print(f"  {'':7}step 0{' ' * (len(sampled) - 12)}step {len(losses) - 1}")


def cmd_train_summary(args: argparse.Namespace) -> None:
    token = load_token(args)
    job_id = args.train_summary
    content = _fetch_log_content(job_id, token, args.base_url)
    if not content.strip():
        print("日志为空（任务可能还在等待中，或尚未产生训练日志）")
        return

    lines = content.splitlines()
    metrics = _parse_metrics(lines)

    print("-" * 50)
    print(f"训练汇报 - {job_id}")
    print("-" * 50)

    if not metrics:
        print("未能解析到 loss 指标，以下是日志末尾 20 行（供参考）：")
        print("\n".join(lines[-20:]))
        return

    losses = [m["loss"] for m in metrics]
    lrs = [m["learning_rate"] for m in metrics if m.get("learning_rate") is not None]

    first_loss, last_loss = losses[0], losses[-1]
    min_loss, max_loss = min(losses), max(losses)
    drop_pct = (first_loss - last_loss) / first_loss * 100 if first_loss > 0 else 0

    print(f"\n[loss 趋势]  共 {len(metrics)} 条记录")
    print(f"  起始：{first_loss:.4f}  ->  最终：{last_loss:.4f}")
    print(f"  最低：{min_loss:.4f}  最高：{max_loss:.4f}")
    if drop_pct > 10:
        verdict = "[OK] 明显下降，训练收敛正常"
    elif drop_pct > 2:
        verdict = "[OK] 轻微下降，可以尝试增加 epochs"
    elif drop_pct > -2:
        verdict = "[!!] 基本持平，建议调小学习率或增加 epochs"
    else:
        verdict = "[!!] loss 上升，训练不稳定，建议检查数据质量和学习率"
    if drop_pct >= 0:
        change_str = f"loss 下降 {drop_pct:.1f}%"
    else:
        change_str = f"loss 上升 {abs(drop_pct):.1f}%"
    print(f"  变化：{change_str}  {verdict}")

    if lrs:
        print(f"\n[学习率]")
        print(f"  起始：{lrs[0]:.2e}  最终：{lrs[-1]:.2e}")
        if lrs[-1] < lrs[0] * 0.5:
            print(f"  趋势：lr 已衰减（正常，调度器生效）")
        elif lrs[-1] > lrs[0]:
            print(f"  趋势：lr 先升（warmup 阶段）")

    # ASCII loss 折线图
    if len(losses) >= 2:
        _print_ascii_loss_chart(losses)

    error_lines = [l for l in lines if re.search(r"\b(error|exception|traceback)\b", l, re.IGNORECASE)]
    if error_lines:
        print(f"\n[!!] 发现 {len(error_lines)} 条错误/异常（最后 3 条）：")
        for l in error_lines[-3:]:
            print(f"  {l[:120]}")

    print()


# ------------------------------- cancel ----------------------- #

def cmd_cancel(args: argparse.Namespace) -> None:
    token = load_token(args)
    job_id = args.cancel

    print(f"取消任务：{job_id}")
    api("POST", f"/v1/train/jobs/{job_id}/cancel", token, args.base_url)
    print("取消请求已发送。")
    print(f"\n确认状态：python3 {Path(__file__).name} --status {job_id}")


# --------------------------- open tensorboard ---------------- #

def cmd_open_tb(args: argparse.Namespace) -> None:
    token = load_token(args)
    job_id = args.open_tb
    data = api("GET", f"/v1/train/jobs/{job_id}", token, args.base_url)
    tb_url = data.get("tensorboardUrl", "")
    if not tb_url:
        state = data.get("state", "")
        die(f"该任务暂无 Tensorboard 地址（当前状态：{state}）。\n训练开始后（running/training）才会生成地址。")
    _open_url(tb_url)


# ---------------------------- eval guide --------------------- #

def cmd_eval_guide(args: argparse.Namespace) -> None:
    path = Path(args.eval_guide)
    if not path.exists():
        die(f"文件不存在：{path}")

    lines = [l.strip() for l in path.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    samples: list[dict] = []
    for line in lines[:200]:
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                samples.append(obj)
        except Exception:
            pass

    if not samples:
        die("未能从文件中解析出有效样本。")

    # 检测格式
    first = samples[0]
    is_ernie = "src" in first
    is_sharegpt = not is_ernie and "conversations" in first and isinstance(first["conversations"], list)

    def _unwrap(val: object) -> str:
        if isinstance(val, list):
            return str(val[0]) if val else ""
        return str(val) if val is not None else ""

    def _get_src(s: dict) -> str:
        if is_ernie:
            return _unwrap(s.get("src", ""))
        if is_sharegpt:
            convs = s.get("conversations", [])
            human = next((c.get("value", "") for c in convs if c.get("from") == "human"), "")
            return str(human)
        return _unwrap(s.get("instruction", ""))

    def _get_tgt(s: dict) -> str:
        if is_ernie:
            return _unwrap(s.get("tgt", ""))
        if is_sharegpt:
            convs = s.get("conversations", [])
            gpt = next((c.get("value", "") for c in convs if c.get("from") == "gpt"), "")
            return str(gpt)
        return _unwrap(s.get("output", ""))

    # 选取最多 5 个有代表性的样本
    random.seed(42)
    picked = random.sample(samples, min(5, len(samples)))

    print(f"\n{chr(61) * 55}")
    print("  训练后效果测试建议（基于你的训练数据生成）")
    print(f"{chr(61) * 55}")
    print(f"\n  数据集：{path.name}，共 {len(samples)} 条样本\n")

    print("【测试问题 1：训练集内问题（验证是否学到）】")
    print("将以下问题发给微调后的模型，看回答是否接近期望输出：\n")
    for i, s in enumerate(picked[:3], 1):
        src = _get_src(s)[:120]
        tgt = _get_tgt(s)[:80]
        print(f"  问题 {i}：{src}")
        print(f"  期望回答（节选）：{tgt}...\n")

    print("\n【泛化测试：换个说法问同类问题】")
    print("把下面这些问题用自己的话改写后再问一遍，看风格是否保持一致：\n")
    for i, s in enumerate(picked[:2], 1):
        src = _get_src(s)[:120]
        print(f"  问题 {i}：{src}\n")

    print("\n【效果判断标准】")
    print("  [OK] 回答风格与训练数据一致 -> 微调有效")
    print("  [OK] 专有名词处理更准确 -> 领域知识已注入")
    print("  [X]  回答完全复制训练样本 -> 可能过拟合，减少 epochs 或增加数据量")
    print("  [X]  回答与基底模型无区别 -> 可能欠拟合，增加 epochs 或检查数据质量")
    print("\n" + "=" * 55 + "\n")


# ----------------------------- main --------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AiStudio 无代码训练 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 通用参数
    p.add_argument("--api-key", metavar="TOKEN", help="Access Token（也可用环境变量 AISTUDIO_ACCESS_TOKEN）")
    p.add_argument("--env-file", metavar="FILE", help="从 .env 文件读取 token")
    p.add_argument("--base-url", default=BASE_URL, metavar="URL", help=f"API 地址（默认 {BASE_URL}）")

    # 子命令
    p.add_argument("--verify-token", action="store_true", help="验证 Access Token 是否有效")
    p.add_argument("--list-models", action="store_true", help="列出平台白名单中所有可用模型")
    p.add_argument("--list-datasets", action="store_true", help="列出内置推荐数据集（不用自己准备数据）")
    p.add_argument("--check-data", metavar="FILE", help="检查数据格式")
    p.add_argument("--suggest-params", metavar="FILE", help="根据数据集推荐超参数")
    p.add_argument("--model-type", choices=["ernie", "llama"], default="ernie", help="模型类型（配合 --suggest-params）")
    p.add_argument("--status", metavar="JOB_ID", help="查看任务状态")
    p.add_argument("--env-check", action="store_true", help="检查本地环境（Python 版本、依赖包）")
    p.add_argument("--poll", metavar="JOB_ID", help="持续轮询进度")
    p.add_argument("--interval", type=int, default=30, metavar="SEC", help="轮询间隔（秒，默认 30）")
    p.add_argument("--logs", metavar="JOB_ID", help="查看训练日志")
    p.add_argument("--system", action="store_true", help="查看系统日志（配合 --logs）")
    p.add_argument("--cancel", metavar="JOB_ID", help="取消任务")
    p.add_argument("--open-tb", metavar="JOB_ID", help="打开 Tensorboard 看板")
    p.add_argument("--eval-guide", metavar="FILE", help="根据训练数据生成测试问题")
    p.add_argument("--train-summary", metavar="JOB_ID", help="训练完成后汇报 loss/lr 趋势")

    # submit 参数
    p.add_argument("--submit", action="store_true", help="提交训练任务")
    p.add_argument("--base-model", metavar="MODEL", help="基底模型（如 PaddlePaddle/ERNIE-4.5-0.3B-PT）")
    p.add_argument("--train-type", metavar="TYPE", help="训练类型（SFT/Full | SFT/LoRA）")
    p.add_argument("--train-data", metavar="USER/DATASET", help="数据集仓库路径（如 myuser/my_dataset）")
    p.add_argument("--train-file", metavar="FILENAME", help="数据文件名（如 train.jsonl）")
    p.add_argument("--params", metavar="JSON", help="超参数 JSON 字符串")
    p.add_argument("--name", metavar="NAME", help="任务名称（只允许字母、数字、下划线）")
    p.add_argument("--description", metavar="DESC", help="任务描述")
    p.add_argument("--output-repo", metavar="USER/REPO", help="模型输出仓库（可选）")
    p.add_argument("--max-run-time", type=int, metavar="HOURS", help="最长运行时间（小时，1-240）")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.env_check:
        cmd_env_check()
    elif args.verify_token:
        cmd_verify_token(args)
    elif args.list_models:
        cmd_list_models(args)
    elif args.list_datasets:
        cmd_list_datasets(args)
    elif args.check_data:
        cmd_check_data(args)
    elif args.suggest_params:
        cmd_suggest_params(args)
    elif args.submit:
        if not args.base_model:
            parser.error("--submit 需要 --base-model")
        if not args.train_type:
            parser.error("--submit 需要 --train-type")
        if not args.train_data:
            parser.error("--submit 需要 --train-data")
        cmd_submit(args)
    elif args.status:
        cmd_status(args)
    elif args.poll:
        cmd_poll(args)
    elif args.logs:
        cmd_logs(args)
    elif args.cancel:
        cmd_cancel(args)
    elif args.open_tb:
        cmd_open_tb(args)
    elif args.eval_guide:
        cmd_eval_guide(args)
    elif args.train_summary:
        cmd_train_summary(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
