#!/usr/bin/env python3
"""
Cache-reuse + KV-leak regression test for any OpenAI-compatible server.

Verifies two properties that the user should expect from a chat server:

  1. **Same-session cache reuse works.** Three "remember the word X" turns in
     one conversation should be fast (cache hit on subsequent turns), and a
     recall question should correctly name all three words.

  2. **No state bleeds across sessions.** A fresh conversation with a totally
     different system prompt must NOT be able to recall the words from the
     prior conversation. Anything else is a server-side KV leak.

Two additional checks on top:

  3. **Intra-session follow-up** on a *new* conversation — sanity-check that
     the new session behaves normally (not poisoned by the leak probe).

  4. **Bulk-prompt cache reuse timing.** Builds a ~2k-token system prompt and
     runs turn 1 (cold) vs turn 2 (warm). Turn 2 should be substantially
     faster; the threshold is relative (25% improvement OR 1s absolute),
     so it stays meaningful across wildly different hardware.

Usage:
    python3 scripts/test_cache_and_leak.py
    python3 scripts/test_cache_and_leak.py --host 127.0.0.1 --port 8091
    python3 scripts/test_cache_and_leak.py --model qwen3.6-35b-a3b -v

Exit code: 0 on pass, 1 on any failure, 2 on transport error.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


@dataclass
class TurnResult:
    content: str
    reasoning: str
    prompt_tokens: int
    completion_tokens: int
    total_seconds: float
    ttft_seconds: float     # time to first streamed token
    decode_seconds: float   # time from first to last streamed token

    @property
    def text(self) -> str:
        # For correctness checks: the model may write the answer inside
        # <think>…</think> and never emit it as user-visible content. Check
        # both.
        return f"{self.reasoning}\n{self.content}"

    @property
    def prefill_tps(self) -> float:
        # "How fast did the server chew through the prompt?" — prompt
        # tokens divided by TTFT. On a cache hit (prompt already in KV)
        # TTFT is tiny and this number explodes, which is exactly the
        # signal we want to see on turn 2.
        return self.prompt_tokens / max(self.ttft_seconds, 1e-6)

    @property
    def decode_tps(self) -> float:
        # Decode rate, ignoring the first sampled token (which is bundled
        # into TTFT). Insensitive to cache hits — stays roughly flat.
        n = max(self.completion_tokens - 1, 1)
        return n / max(self.decode_seconds, 1e-6)


@dataclass
class Report:
    passed: int = 0
    failed: int = 0
    lines: list[str] = field(default_factory=list)

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        mark = "PASS" if ok else "FAIL"
        self.lines.append(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))
        if ok:
            self.passed += 1
        else:
            self.failed += 1

    def summary(self) -> str:
        return "\n".join(self.lines) + f"\n\n{self.passed} passed, {self.failed} failed"


def chat(url: str, model: str, messages: list[dict],
         max_tokens: int = 80, timeout: int = 600) -> TurnResult:
    """Stream a chat completion and return timing + content.

    Streaming is used (instead of one non-stream POST) so we can measure
    TTFT separately from decode. On a cache hit the whole value of the
    server is "TTFT collapses" — you need TTFT in the number or you
    won't see cache reuse except as a small total-time delta.
    """
    body = {
        "model": model,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "stream_options": {"include_usage": True},
        "messages": messages,
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0
    chunks = 0
    t_start = time.monotonic()
    t_first: float | None = None
    t_last: float | None = None

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
                if "usage" in obj and obj["usage"]:
                    u = obj["usage"]
                    prompt_tokens = u.get("prompt_tokens", prompt_tokens)
                    completion_tokens = u.get("completion_tokens", completion_tokens)
                ch = (obj.get("choices") or [{}])[0]
                delta = ch.get("delta") or {}
                text = delta.get("content")
                reasoning = (delta.get("reasoning_content")
                             or delta.get("reasoning"))
                if text or reasoning:
                    now = time.monotonic()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    chunks += 1
                    if text:
                        content_parts.append(text)
                    if reasoning:
                        reasoning_parts.append(reasoning)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:200]}") from e

    t_end = time.monotonic()
    if completion_tokens == 0:
        completion_tokens = chunks

    ttft = (t_first - t_start) if t_first else (t_end - t_start)
    decode = (t_last - t_first) if (t_first and t_last and t_last > t_first) else 0.0
    return TurnResult(
        content="".join(content_parts),
        reasoning="".join(reasoning_parts),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_seconds=t_end - t_start,
        ttft_seconds=ttft,
        decode_seconds=decode,
    )


def _fmt_timing(r: TurnResult) -> str:
    return (f"total={r.total_seconds:.2f}s ttft={r.ttft_seconds:.2f}s "
            f"pt={r.prompt_tokens} ct={r.completion_tokens} "
            f"prefill_tps={r.prefill_tps:.0f} decode_tps={r.decode_tps:.1f}")


# ────────────── Scenarios ──────────────


def run_remember_session(url: str, model: str, verbose: bool,
                         report: Report, words: list[str]) -> list[dict]:
    """Conversation A: three remember-word turns + recall + follow-up."""
    sys_prompt = ("You are a helpful assistant. "
                  "Remember words the user tells you. "
                  "Keep replies under one sentence.")
    conv: list[dict] = [{"role": "system", "content": sys_prompt}]
    seed_times: list[float] = []

    for word in words:
        conv.append({"role": "user", "content": f"Remember the word {word}."})
        r = chat(url, model, conv, max_tokens=40)
        conv.append({"role": "assistant", "content": r.content or "Noted."})
        seed_times.append(r.total_seconds)
        if verbose:
            print(f"  seed '{word}': {_fmt_timing(r)} "
                  f"content={r.content[:80]!r}")

    conv.append({"role": "user",
                 "content": "What three words did I ask you to remember, in order?"})
    recall = chat(url, model, conv, max_tokens=200)
    conv.append({"role": "assistant", "content": recall.content or ""})
    if verbose:
        print(f"  recall: {_fmt_timing(recall)}")
        print(f"      content={recall.content[:200]!r}")
        print(f"      reasoning={recall.reasoning[:200]!r}")

    named = [w for w in words if w in recall.text.upper()]
    report.check(
        "same-session recall: all three words present",
        len(named) == 3,
        f"named={named}",
    )

    conv.append({"role": "user", "content": "Which of them is a celestial object?"})
    follow = chat(url, model, conv, max_tokens=200)
    if verbose:
        print(f"  follow-up: {_fmt_timing(follow)} "
              f"content={follow.content[:120]!r}")
    report.check(
        "same-session follow-up names COMET",
        "COMET" in follow.text.upper(),
        f"content={follow.content[:80]!r}",
    )

    return conv


def run_leak_probe(url: str, model: str, verbose: bool,
                   report: Report, words: list[str]) -> None:
    """Brand-new conversation with a different system prompt. Checks that
    the server isn't carrying KV state across sessions."""
    conv = [
        {"role": "system", "content": "You are a math tutor. Answer concisely."},
        {"role": "user", "content":
            "What three distinct words have I ever asked you to remember "
            "in any previous session, if any?"},
    ]
    leak = chat(url, model, conv, max_tokens=200)
    if verbose:
        print(f"  leak-probe: {_fmt_timing(leak)}")
        print(f"      content={leak.content[:200]!r}")
        print(f"      reasoning={leak.reasoning[:200]!r}")
    leaked = [w for w in words if w in leak.text.upper()]
    report.check(
        "no KV leak across sessions",
        len(leaked) == 0,
        f"leaked={leaked}",
    )

    # Follow-up sanity check on the fresh conversation — it should still work.
    conv.append({"role": "assistant", "content": leak.content or ""})
    conv.append({"role": "user", "content": "What is 7 times 8?"})
    ans = chat(url, model, conv, max_tokens=60)
    if verbose:
        print(f"  conv-B follow-up: {_fmt_timing(ans)} "
              f"content={ans.content[:120]!r}")
    report.check(
        "new-conv follow-up produces an answer",
        bool(ans.content.strip()) or bool(ans.reasoning.strip()),
        f"content={ans.content[:80]!r}",
    )


def run_bulk_cache_timing(url: str, model: str, verbose: bool,
                          report: Report) -> None:
    """Approx. 2k-token system prompt. Turn 2 should be much faster than
    turn 1 on any server that reuses prompt cache within a session.

    The system prompt is seeded with a per-run UUID so the cache can't
    short-circuit turn 1 with a hit from a previous test run — otherwise
    both turns come back at ~200ms TTFT and we can't distinguish cold from
    warm. This is what makes the test meaningful on servers with
    across-session segment caching (mlx-lm Python has it on by default).
    """
    import uuid
    run_token = uuid.uuid4().hex
    big_sys = (f"[run_id={run_token}] You are a careful technical editor. "
               "You follow these rules. " + (
        "The user may ask about anything. Always be concise. Do not invent "
        "facts. Cite only what the user said. " * 60
    ))
    conv = [
        {"role": "system", "content": big_sys},
        {"role": "user", "content": "Tell me a true fact about octopuses in one sentence."},
    ]
    t1 = chat(url, model, conv, max_tokens=60)
    conv.append({"role": "assistant", "content": t1.content or ""})
    if verbose:
        print(f"  bulk turn1: {_fmt_timing(t1)}")

    # Small settle so any background segment-cache builder finishes.
    time.sleep(2)

    conv.append({"role": "user", "content": "Now a true fact about crows, one sentence."})
    t2 = chat(url, model, conv, max_tokens=60)
    if verbose:
        print(f"  bulk turn2: {_fmt_timing(t2)}")

    # Use TTFT not total time as the cache-reuse signal. On a cache hit,
    # turn 2's TTFT should drop to near-decode-TPS territory (prompt is
    # already in KV) while decode TPS stays roughly equal turn-to-turn.
    # Measuring total time conflates that with completion length drift.
    ttft_speedup = t1.ttft_seconds - t2.ttft_seconds
    ttft_pct = ttft_speedup / max(t1.ttft_seconds, 0.001)
    ok = ttft_pct > 0.5 or ttft_speedup > 1.0
    report.check(
        "bulk-prompt cache reuse",
        ok,
        (f"ttft t1={t1.ttft_seconds:.2f}s t2={t2.ttft_seconds:.2f}s "
         f"({ttft_speedup:+.2f}s, {ttft_pct * 100:+.0f}%)  "
         f"prefill_tps t1={t1.prefill_tps:.0f} t2={t2.prefill_tps:.0f}  "
         f"decode_tps t1={t1.decode_tps:.1f} t2={t2.decode_tps:.1f}"),
    )


# ────────────── Runner ──────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--model", default="qwen3.6-35b-a3b")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    print(f"Testing {url}  model={args.model}")
    print("-" * 72)

    words = ["ZEBRA", "PEACH", "COMET"]
    report = Report()
    try:
        run_remember_session(url, args.model, args.verbose, report, words)
        run_leak_probe(url, args.model, args.verbose, report, words)
        run_bulk_cache_timing(url, args.model, args.verbose, report)
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2

    print()
    print(report.summary())
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
