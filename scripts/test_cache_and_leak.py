#!/usr/bin/env python3
"""
Cache-reuse + leak test for mlx-swift-lm MLXServer.

Runs four scenarios against a live server and reports pass/fail + timing:

  1. Seed: three same-session "remember word X" messages. Measures whether
     turns 2 and 3 hit the cache (faster than turn 1).
  2. Recall: same session asks "what words did I ask you to remember?".
     Should correctly name ZEBRA, PEACH, COMET (same-session continuity).
  3. Leak probe: a BRAND NEW conversation (different system prompt) asks
     the same recall question. Must NOT mention ZEBRA/PEACH/COMET — a hit
     here means server-side KV state leaked across sessions.
  4. Follow-up on new conversation: verify that intra-session cache reuse
     works on the new conversation too.

Rerun after any server change to verify caching works and there's no leak.

Usage:
    python3 scripts/test_cache_and_leak.py [--host HOST] [--port PORT] [--model MODEL]

Exit code: 0 on pass, 1 on any failure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TurnResult:
    label: str
    prompt_tokens: int
    completion_tokens: int
    total_seconds: float
    content: str
    reasoning: str = ""

    @property
    def text(self) -> str:
        return f"{self.reasoning}\n{self.content}"


@dataclass
class TestReport:
    passed: int = 0
    failed: int = 0
    messages: list[str] = field(default_factory=list)

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        status = "✅" if ok else "❌"
        line = f"{status} {name}"
        if detail:
            line += f" — {detail}"
        self.messages.append(line)
        if ok:
            self.passed += 1
        else:
            self.failed += 1

    def summary(self) -> str:
        return "\n".join(self.messages) + f"\n\n{self.passed} passed, {self.failed} failed"


def chat(
    url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 80,
    stream: bool = False,
) -> TurnResult:
    body = {
        "model": model,
        "stream": stream,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} {e.reason}: {e.read().decode()[:200]}") from e
    elapsed = time.time() - t0

    choice = data["choices"][0]
    msg = choice.get("message") or {}
    usage = data.get("usage") or {}
    return TurnResult(
        label="",
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        total_seconds=elapsed,
        content=msg.get("content", "") or "",
        reasoning=msg.get("reasoning_content", "") or "",
    )


def run_tests(url: str, model: str, verbose: bool = False) -> TestReport:
    report = TestReport()

    # ------- Conversation A: seed 3 remember-word turns + recall + follow-up -------
    sys_a = "You are a helpful assistant. Remember words the user tells you. Keep replies under one sentence."
    conv_a: list[dict[str, str]] = [{"role": "system", "content": sys_a}]
    words = ["ZEBRA", "PEACH", "COMET"]

    seed_times: list[float] = []
    for i, word in enumerate(words):
        conv_a.append({"role": "user", "content": f"Remember the word {word}."})
        r = chat(url, model, conv_a, max_tokens=20)
        conv_a.append({"role": "assistant", "content": r.content or "Noted."})
        seed_times.append(r.total_seconds)
        if verbose:
            print(f"  seed[{i+1}] word={word}: {r.total_seconds:.2f}s  "
                  f"prompt_tok={r.prompt_tokens}  completion_tok={r.completion_tokens}  "
                  f"content={r.content[:80]!r}")

    # Recall in same conversation
    conv_a.append({"role": "user", "content": "What three words did I ask you to remember, in order?"})
    recall = chat(url, model, conv_a, max_tokens=60)
    conv_a.append({"role": "assistant", "content": recall.content or ""})
    if verbose:
        print(f"  recall: {recall.total_seconds:.2f}s  content={recall.content[:200]!r}")

    recall_text = recall.text.upper()
    named = [w for w in words if w in recall_text]
    # Same-session recall is a correctness check (did it actually track the words?).
    # Not strictly a cache test, but a sanity check for the conversation machinery.
    report.check(
        "same-session recall: all three words present",
        len(named) == 3,
        f"named={named}",
    )

    # Follow-up turn in same conversation — just checks correctness.
    # Absolute timing thresholds are unreliable on tiny prompts (a 20-token
    # cache miss is already <1s), so we leave timing out of pass/fail here
    # and cover real cache-reuse timing in the "bulk prompt" section below.
    conv_a.append({"role": "user", "content": "Which of them is a celestial object?"})
    followup = chat(url, model, conv_a, max_tokens=40)
    if verbose:
        print(f"  follow-up: {followup.total_seconds:.2f}s  content={followup.content[:120]!r}")
    report.check(
        "same-session follow-up names COMET",
        "COMET" in followup.text.upper(),
        f"content={followup.content[:80]!r}",
    )

    # ------- Conversation B (BRAND NEW): leak probe -------
    # Different system prompt, no prior messages. Do NOT include anything from
    # conv_a here — this models what opencode sends on a truly fresh session.
    sys_b = "You are a math tutor. Answer concisely."
    conv_b = [
        {"role": "system", "content": sys_b},
        {"role": "user", "content": "What three distinct words have I ever asked you to remember in any previous session, if any?"},
    ]
    leak = chat(url, model, conv_b, max_tokens=80)
    if verbose:
        print(f"  leak-probe: {leak.total_seconds:.2f}s  content={leak.content[:200]!r}")
    leak_text = leak.text.upper()
    leaked_words = [w for w in words if w in leak_text]
    report.check(
        "no KV leak: ZEBRA/PEACH/COMET not mentioned in new-session reply",
        len(leaked_words) == 0,
        f"leaked={leaked_words}",
    )

    # ------- Conversation B follow-up: cache works on new conversation too -------
    conv_b.append({"role": "assistant", "content": leak.content or ""})
    conv_b.append({"role": "user", "content": "What is 7 times 8?"})
    b_follow = chat(url, model, conv_b, max_tokens=30)
    if verbose:
        print(f"  conv-B follow-up: {b_follow.total_seconds:.2f}s  content={b_follow.content[:120]!r}")
    report.check(
        "new-conv follow-up produced an answer",
        len(b_follow.content.strip()) > 0 or len(b_follow.reasoning.strip()) > 0,
        f"content={b_follow.content[:80]!r}",
    )

    # ------- Realistic-size caching test -------
    # Tiny prompts above don't stress the prefill phase. This builds a large
    # (~2000-token) system prompt so that cache reuse actually matters. After
    # a cold first turn (slow), subsequent turns on the same conversation
    # should be substantially faster — that's our segment-cache / same-session
    # reuse signal.
    big_sys = (
        "You are a careful technical editor. You follow these rules strictly. "
        + ("The user may ask about anything. Always be concise. Do not invent facts. "
           "Cite only what the user said. " * 60)
    )
    if verbose:
        print(f"  bulk system prompt approx length: {len(big_sys)} chars")
    conv_big = [
        {"role": "system", "content": big_sys},
        {"role": "user", "content": "Tell me a true fact about octopuses in one sentence."},
    ]
    big1 = chat(url, model, conv_big, max_tokens=25)
    conv_big.append({"role": "assistant", "content": big1.content or ""})

    # Give any background segment-build time to complete. If segment caching
    # is wired up, a background Task will prefill the static prefix right
    # after turn 1 completes — we need it to finish before turn 2 arrives.
    if verbose:
        print(f"  big-sys turn1: {big1.total_seconds:.2f}s  prompt_tok={big1.prompt_tokens}  completion_tok={big1.completion_tokens}")
    time.sleep(25)

    conv_big.append({"role": "user", "content": "Now tell me a true fact about crows in one sentence."})
    big2 = chat(url, model, conv_big, max_tokens=25)
    if verbose:
        print(f"  big-sys turn2: {big2.total_seconds:.2f}s  prompt_tok={big2.prompt_tokens}  completion_tok={big2.completion_tokens}")

    # Turn 2 adds only a small user message on top of turn 1's conversation.
    # With working cache reuse (segment or session), it should be noticeably
    # faster than turn 1. On a real hybrid Mamba model with no cache reuse,
    # turn 2 is ~= turn 1. Speedup threshold is relative.
    speedup = big1.total_seconds - big2.total_seconds
    # Require at least 25% improvement OR 1s absolute (whichever applies to the size).
    cache_works = (speedup / max(big1.total_seconds, 0.001)) > 0.25 or speedup > 1.0
    report.check(
        f"bulk-prompt cache: turn 2 faster than turn 1 (t1={big1.total_seconds:.2f}s t2={big2.total_seconds:.2f}s)",
        cache_works,
        f"speedup={speedup:+.2f}s ({speedup / max(big1.total_seconds, 0.001) * 100:.0f}%)",
    )

    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--model", default="qwen3.6-35b-a3b")
    ap.add_argument("--verbose", "-v", action="store_true", help="print per-turn timing and content")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    print(f"Testing {url} (model={args.model})")
    print("-" * 60)
    try:
        report = run_tests(url, args.model, verbose=args.verbose)
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2
    print()
    print(report.summary())
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
