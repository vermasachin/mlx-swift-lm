# Baseline environment

All TPS and cache-behavior numbers in this directory were captured against
this environment. Before attributing a regression to a code change, confirm
the environment here still matches.

## Hardware / OS
- `uname -a` → Darwin Sachins-MacBook-Pro.local 25.4.0 Darwin Kernel Version 25.4.0: Thu Mar 19 19:33:25 PDT 2026; root:xnu-12377.101.15~1/RELEASE_ARM64_T6041 arm64
- `sw_vers`:
  - ProductName:		macOS
  - ProductVersion:		26.4.1
  - BuildVersion:		25E253
- Memory: 48 GB
- CPU: Apple M4 Pro

## Server used for baseline
- mlx-lm **Python** server (upstream `ml-explore/mlx-lm`, installed via pip)
- Version: 0.31.2
- MLX: n/a
- Launched with:
  ```
  .venv/bin/python -m mlx_lm server \
      --model /Users/sachinverma/personal/models/mlx/Qwen3.6-35B-A3B-UD-MLX-4bit \
      --port 8091 --host 127.0.0.1
  ```

## Model
- Path: `/Users/sachinverma/personal/models/mlx/Qwen3.6-35B-A3B-UD-MLX-4bit`
- chat_template.jinja md5: a7f294a5f0be5f1903214304f259f87f (matches `unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit` on HuggingFace, byte-identical)
- architectures / model_type (from config.json):
  - model_type: qwen3_5_moe
  - architectures: ['Qwen3_5MoeForConditionalGeneration']

## Sampling parameters (same across all benchmarks)
Taken from Unsloth's Qwen3.6 "thinking + coding" recommended settings
(https://unsloth.ai/docs/models/qwen3.6):
- `temperature: 0.6`
- `top_p: 0.95`
- `top_k: 20`
- `min_p: 0.0`
- `presence_penalty: 0.0`
