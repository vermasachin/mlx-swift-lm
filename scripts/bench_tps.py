#!/usr/bin/env python3
"""
TPS benchmark for MLXServer (or any OpenAI-compatible chat endpoint).

Measures *separately*:
  - prefill TPS = prompt_tokens / (time-to-first-token)
  - decode  TPS = (completion_tokens - 1) / (time_last_chunk - time_first_chunk)

These two numbers answer different questions. Prefill TPS is how fast the
server chews through the prompt before it starts producing output. Decode TPS
is how fast it generates each subsequent token — what a user feels during
streaming. Conflating them (e.g. dividing completion_tokens by total time)
pollutes the number with whichever phase was bigger, which is why naive
"tps" readings are so unstable across prompt sizes.

Scenarios
---------

1. `short_long` — tiny prompt, long generation. Isolates pure decode.
2. `long_short` — ~2k-token prompt, short generation. Prefill-dominated.
3. `opencode_like` — ~14k-token system prefix + one user message. Matches
   the real opencode workload so we can compare against production.
4. `multiturn` — 5-turn back-and-forth, each turn builds on the prior
   assistant reply. Cold turn 1; turns 2+ should benefit from prompt-cache
   reuse. Per-turn prefill TPS should climb turn-over-turn if cache works;
   decode TPS should stay flat.

Each scenario runs REPEATS times and reports median/min/max.

Usage:
    python3 scripts/bench_tps.py
    python3 scripts/bench_tps.py --scenario opencode_like --repeats 3
    python3 scripts/bench_tps.py --host 127.0.0.1 --port 8091 --model qwen3.6-35b-a3b
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable


@dataclass
class TurnTiming:
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float           # time to first token (prefill + a bit)
    decode_s: float         # time from first to last chunk
    total_s: float

    @property
    def prefill_tps(self) -> float:
        # Prefill TPS: how fast the server chewed the prompt. TTFT includes
        # the first token sample, but for large prompts TTFT ≈ prefill time.
        return self.prompt_tokens / max(self.ttft_s, 1e-6)

    @property
    def decode_tps(self) -> float:
        # Decode TPS: per-token generation rate. Exclude the first token
        # (which is bundled into prefill) so this is a pure decode measurement.
        n = max(self.completion_tokens - 1, 1)
        return n / max(self.decode_s, 1e-6)


def stream_chat(
    url: str, model: str, messages: list[dict], max_tokens: int,
    timeout: int = 600,
) -> TurnTiming:
    """POST streaming; time first token + subsequent chunks separately."""
    body = {
        "model": model,
        "stream": True,
        "max_tokens": max_tokens,
        # Sampling params from Unsloth's Qwen3.6 "thinking + coding" guide:
        # https://unsloth.ai/docs/models/qwen3.6 — this is the regime the
        # model was tuned for, so decode-TPS numbers measured here match
        # what a real coding-assistant workload (e.g. opencode) will see.
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        # stream_options.include_usage is the OpenAI-standard way to ask the
        # server to emit a final chunk with prompt_tokens/completion_tokens.
        # mlx-lm's Python server requires this explicit opt-in; Swift-side
        # servers generally include it by default but the flag is harmless.
        "stream_options": {"include_usage": True},
        "messages": messages,
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t_start = time.monotonic()
    t_first: float | None = None
    t_last: float | None = None
    completion_tokens = 0
    prompt_tokens = 0
    chunks = 0

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                # Usage block lands on the final chunk for most OpenAI-compat
                # servers; capture it whenever present.
                if "usage" in obj and obj["usage"]:
                    u = obj["usage"]
                    prompt_tokens = u.get("prompt_tokens", prompt_tokens)
                    completion_tokens = u.get("completion_tokens", completion_tokens)
                ch = (obj.get("choices") or [{}])[0]
                delta = ch.get("delta") or {}
                # `reasoning_content` is the OpenAI-ish name our Swift server
                # uses; mlx-lm's Python server calls the same field `reasoning`.
                # Treat both as decoded text for TPS accounting.
                text = (delta.get("content")
                        or delta.get("reasoning_content")
                        or delta.get("reasoning"))
                if text:
                    now = time.monotonic()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    chunks += 1
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:200]}") from e

    t_end = time.monotonic()

    # Fallback if server didn't report usage in streaming.
    if completion_tokens == 0:
        completion_tokens = chunks

    ttft = (t_first - t_start) if t_first else (t_end - t_start)
    decode = (t_last - t_first) if (t_first and t_last and t_last > t_first) else 0.0
    return TurnTiming(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        ttft_s=ttft,
        decode_s=decode,
        total_s=t_end - t_start,
    )


# --------------------------- Scenarios ---------------------------

FILLER = ("The quick brown fox jumps over the lazy dog. "
          "Pack my box with five dozen liquor jugs. ")


def scenario_short_long(url, model) -> list[TurnTiming]:
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a 300-word story about a lighthouse keeper."},
    ]
    return [stream_chat(url, model, msgs, max_tokens=400)]


def scenario_long_short(url, model) -> list[TurnTiming]:
    # ~2000-token system prompt.
    long_sys = "You are an assistant. Background: " + (FILLER * 200)
    msgs = [
        {"role": "system", "content": long_sys},
        {"role": "user", "content": "In one sentence: what color is the sky?"},
    ]
    return [stream_chat(url, model, msgs, max_tokens=60)]


def scenario_opencode_like(url, model) -> list[TurnTiming]:
    # Approximate opencode's large system+tools baseline with a ~14k-token
    # filler. Matches the real production prompt size.
    huge_sys = "You are a coding assistant. " + (FILLER * 1500)
    msgs = [
        {"role": "system", "content": huge_sys},
        {"role": "user", "content": "Say hello in one short sentence."},
    ]
    return [stream_chat(url, model, msgs, max_tokens=60)]


def scenario_multiturn(url, model) -> list[TurnTiming]:
    """
    5-turn conversation. Each turn's prompt includes all prior assistant
    replies, so context grows monotonically. With working prompt-cache
    reuse, prefill TPS on turn 2+ should be much higher than turn 1
    (server reuses prior KV state and only prefills the delta).
    """
    sys = "You are a concise assistant. Keep answers under 40 words."
    msgs = [{"role": "system", "content": sys}]
    questions = [
        "What is the capital of France?",
        "And what language do they speak there?",
        "Name one famous landmark.",
        "What year was it built?",
        "Who designed it?",
    ]
    timings: list[TurnTiming] = []
    for q in questions:
        msgs.append({"role": "user", "content": q})
        t = stream_chat(url, model, msgs, max_tokens=50)
        timings.append(t)
        # We don't get the assistant reply text back from the timing object,
        # but for cache-reuse measurement the *tokens* the server saved are
        # what matter; we can stand in with a stub reply. This makes turn N+1
        # diverge slightly from turn N's saved cache (by a few tokens), which
        # is actually more realistic — matches how opencode re-renders past
        # assistant turns through the chat template.
        msgs.append({"role": "assistant", "content": "(prior reply)"})
    return timings


SCENARIOS: dict[str, Callable] = {
    "short_long": scenario_short_long,
    "long_short": scenario_long_short,
    "opencode_like": scenario_opencode_like,
    "multiturn": scenario_multiturn,
}


# --------------------------- Runner ---------------------------

def summarize(label: str, runs: list[TurnTiming]) -> str:
    prefill_tps = [r.prefill_tps for r in runs]
    decode_tps = [r.decode_tps for r in runs]
    pt = runs[0].prompt_tokens
    ct = round(statistics.mean(r.completion_tokens for r in runs))
    return (
        f"{label:<28} pt={pt:>5}  ct~{ct:>4}  "
        f"prefill_tps={statistics.median(prefill_tps):>7.1f} "
        f"(min={min(prefill_tps):.0f} max={max(prefill_tps):.0f})  "
        f"decode_tps={statistics.median(decode_tps):>6.1f} "
        f"(min={min(decode_tps):.1f} max={max(decode_tps):.1f})"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--model", default="qwen3.6-35b-a3b")
    ap.add_argument("--scenario", choices=list(SCENARIOS) + ["all"], default="all")
    ap.add_argument("--repeats", type=int, default=3,
                    help="how many times to run each scenario")
    ap.add_argument("--warmup", action="store_true", default=True,
                    help="send a throwaway request first to load kernels (default on)")
    ap.add_argument("--label", default=None,
                    help="tag written into the results file (e.g. 'python-mlx-lm-0.31.2')")
    ap.add_argument("--out", default=None,
                    help="results file (default: bench-results/<timestamp>-<label>.txt)")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/v1/chat/completions"

    # Set up results file. Tee stdout so the user sees the report live AND we
    # get a persisted record for side-by-side comparison between runs.
    import os
    import datetime
    label = args.label or "run"
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = args.out or f"bench-results/{stamp}-{label}.txt"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_fh = open(out_path, "w")

    def say(s: str = "") -> None:
        print(s)
        print(s, file=out_fh, flush=True)

    say(f"Benchmarking {url}  model={args.model}  repeats={args.repeats}")
    say(f"Label: {label}   Results: {out_path}")
    say("-" * 100)

    if args.warmup:
        try:
            stream_chat(url, args.model,
                        [{"role": "user", "content": "hi"}], max_tokens=5)
        except Exception as e:
            print(f"warmup failed: {e}", file=sys.stderr)
            return 2

    scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]

    for name in scenarios:
        all_runs: list[TurnTiming] = []
        per_turn: list[list[TurnTiming]] = []
        for _ in range(args.repeats):
            runs = SCENARIOS[name](url, args.model)
            all_runs.extend(runs)
            per_turn.append(runs)

        if name == "multiturn":
            # Print each turn separately — cache reuse is visible turn-by-turn.
            say(f"\n[multiturn] — reuse should lift prefill_tps on turns 2+")
            nturns = len(per_turn[0])
            for t in range(nturns):
                turn_runs = [rep[t] for rep in per_turn]
                say("  " + summarize(f"turn {t+1}", turn_runs))
                for i, r in enumerate(turn_runs, 1):
                    say(f"      run{i}: pt={r.prompt_tokens} ct={r.completion_tokens} "
                        f"prefill_tps={r.prefill_tps:.1f} decode_tps={r.decode_tps:.1f}")
        else:
            say(summarize(name, all_runs))
            for i, r in enumerate(all_runs, 1):
                say(f"    run{i}: pt={r.prompt_tokens} ct={r.completion_tokens} "
                    f"prefill_tps={r.prefill_tps:.1f} decode_tps={r.decode_tps:.1f}")

    out_fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
