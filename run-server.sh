#!/usr/bin/env bash
# Launch MLXServer under pm2 (or stand-alone). Wrapper so pm2 has a stable
# entry point independent of swift build output paths.
#
# Environment overrides:
#   MODEL      path or HF id of the model to load
#   PORT       listen port (default 8091)
#   SLOTS      parallel inference slots (default 1; phase-2 testing ran at 1)
#   KV_SCHEME  optional KV quant scheme, e.g. turbo4v2, turbo4, affine4
#
# Usage (outside pm2):
#   ./run-server.sh
#   PORT=8080 MODEL=/path/to/model ./run-server.sh
#
# Binary comes from `swift build -c release --product MLXServer`. We point
# directly at the architecture-specific path SPM produces; the symlinked
# `.build/release/` alias isn't always present.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$REPO_ROOT/.build/arm64-apple-macosx/release/MLXServer"

# mlx-community uniform-ish 4-bit quant (loads with the per-layer-quant
# fix in commit 11fc00a). Swap with Unsloth UD-MLX-4bit or the 8-bit
# uniform variant by setting MODEL= before launching.
MODEL="${MODEL:-/Users/sachinverma/personal/models/mlx/Qwen3.6-35B-A3B-4bit-mlxc}"
PORT="${PORT:-8091}"
SLOTS="${SLOTS:-1}"

if [[ ! -x "$BIN" ]]; then
    echo "MLXServer binary not found at $BIN" >&2
    echo "Build it first:  swift build -c release --product MLXServer" >&2
    exit 1
fi

ARGS=(--model "$MODEL" --port "$PORT" --slots "$SLOTS")
if [[ -n "${KV_SCHEME:-}" ]]; then
    ARGS+=(--kv "$KV_SCHEME")
fi

exec "$BIN" "${ARGS[@]}"
