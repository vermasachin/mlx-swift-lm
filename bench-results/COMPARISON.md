# Baseline comparison table

Columns grow left-to-right as we add phases. Never overwrite an existing
column — the whole point is to see every step of the ladder.

See `ENVIRONMENT.md` for hardware/software details.

## Decode TPS (higher is better — user-perceived streaming speed)

| scenario                 | ct   | Python, bf16 KV | Python, 4-bit KV | Swift phase0 (alpha)     | Swift phase1 (PR #62) | Swift phase1b (+usage fix) |
|--------------------------|-----:|----------------:|-----------------:|-------------------------:|----------------------:|---------------------------:|
| short prompt, long gen   |  400 |            65.2 |             65.4 |          70.6 ✨         |            64.0       |            62.9            |
| long prompt, short gen   |   60 |            61.5 |             61.7 |  66.5 → ~8 ⚠️            |            58.5  ✅   |            63.3  ✅        |
| opencode-like (~30k pt)  |   60 |            47.0 |             46.7 |  53.2 → ~20 ⚠️           |         48-128 (⚠️ noisy) |     48 → 32 ⚠️             |
| multiturn turn 1         |   50 |            64.7 |             63.8 |  ~8 ⚠️ (always 2nd+ req) |            46.0  ✅   |            33.7  ⚠️        |
| multiturn turn 5         |   50 |            64.9 |             65.4 |  ~8 ⚠️                   |            34.4  ⚠️   |            32.3  ⚠️        |

**Phase 1b finding:** the token-count fix clears up observability but
multiturn decode is flat at ~33 tps across all 5 turns — which is
actually more useful data than phase 1's 46→34 (phase 1's "degradation"
was partly an artifact of short completions inflating early turns).
*Any* multi-turn request decodes at ~33 tps; any first-turn-with-short-
prompt decodes at ~63. That's about a **2× decode gap vs Python** for
the exact multi-turn workload opencode drives. Investigation is issue
#1 from the research phase — likely in `ServerCache.fetch()` copy-on-hit
or some per-turn fixed cost. That's the next thing to profile.

**Phase 1 finding:** PR #62 fixes the sequential-request decode collapse
from phase 0 (small context, stable TPS ≥ 58 across runs now). But it's
slightly below Python on short gens (64 vs 65) and drops ~34 tps by the
5th turn of a multiturn conversation — some in-session degradation
remains. Also saw usage-token-count bugs on warm-cache turns (server
reporting pt=33 for a 1600-token prompt that actually got prefilled).

**Phase 0 finding:** on a cold server, the Swift library beats Python
(70.6 tps vs 65.2 on short-prompt decode). **Every sequential request
after the first collapses to 6–10 tps** regardless of scenario. This
matches the symptom described in open PR #62 on ekryski's alpha:
*"Sequential request fix — second request no longer hangs"* and
*"Streaming fixes — Connection: close for SSE, proper socket cleanup
after [DONE]"*. The Swift library itself is competitive; only the
currently-merged MLXServer has a sequential-request bug that PR #62
already claims to fix. Next phase: evaluate/apply PR #62.

Key finding: **KV quantization barely moves decode TPS on this model.**
That's because this is a hybrid Mamba model (10 attention + 30 Mamba
layers). Attention KV is a minority of the memory traffic; most of the
work is Mamba state + MoE weights. Useful to remember when reasoning
about what a Swift TurboQuant port will and won't help.

## Prefill TPS (higher is better — affects TTFT)

| scenario               | pt    | Python, bf16 KV       | Python, 4-bit KV      | Swift phase0 (alpha)      | Swift phase1 (PR #62) |
|------------------------|------:|----------------------:|----------------------:|--------------------------:|----------------------:|
| short prompt           |    36 |  247 (first 123)      |  254 (first 124)      | 165 (first 114)           | 200 (stable)          |
| long prompt 4k         |  4032 | 24k (first 748)       | 27k (first 750)       | 5.7k (first 185)          | 573 (first 646)       |
| opencode-like 30k      | 30028 | 132k (first 595)      | 134k (first 595)      | 31k (first 156)           | 181 (first 446)       |
| multiturn turn 5       |   115 | 507                   | 490                   |  126                      | 122                   |

**Phase 1 prefill:** sequential-run cache reuse behavior that Python has
(huge numbers on warm runs via segment cache) isn't present — phase 1's
Swift numbers are all roughly cold-prefill rates. The server isn't yet
doing cross-session prefix caching. Cold numbers: Python's 595 tps on
30k → Swift 446 tps (0.75×) — much closer to parity than phase 0 was.

"First" = cold (no segment-cache hit yet). The higher numbers on runs 2+
are server-side prompt-cache hits (both servers do this). Swift phase0's
"first" is consistently ~2–4× slower than Python on cold prefill, which
is a separate issue from the sequential-request decode collapse above.

## Cache / leak checks (bench-results/cache-leak-*.txt)

| check                              | py bf16 | py kv4 | swift phase0 (alpha) | swift phase1 (PR #62) | swift phase1b |
|------------------------------------|:-------:|:------:|:--------------------:|:---------------------:|:-------------:|
| same-session recall (3 words)      |   ✅    |   ✅   |         ✅           |          ✅           |      ✅       |
| same-session follow-up names COMET |   ✅    |   ✅   |         ✅           |          ✅           |      ✅       |
| no KV leak across sessions         |   ✅    |   ✅   |         ✅           |          ✅           |      ✅       |
| new-conv follow-up still works     |   ✅    |   ✅   |         ✅           |          ✅           |      ✅       |
| bulk-prompt cache reuse (cold→warm)|   ✅    |   ✅   |         ❌           |          ✅           |      ✅       |
| cold TTFT on ~1500-token prompt    |  2.13s  |  2.19s |       17.40s         |        11.55s         |    13.45s     |
| warm TTFT (same prompt, turn 2)    |  0.55s  |  0.48s |       26.25s         |         0.52s ✨      |    0.96s  ✅  |
| warm prefill_tps                   |  2781   |  3206  |          61          |     64 (counting bug) |    1658  ✅   |
| warm `cached_tokens` reported      |  n/a    |  n/a   |        n/a           |        n/a            |  1583/1596    |

(cache-reuse uses a per-run UUID in the system prompt so nothing from
earlier runs can short-circuit turn 1.)

**Phase 0 cache-reuse reality**: turn 2 is actually *slower* than turn 1
(TTFT 26s vs 17s). That confirms the sequential-request degradation —
every request on a warm server is already in the hung/slow-decode path,
and turn 2 accumulates more of it. The no-KV-leak pass is a true pass:
whatever the server is doing wrong, it isn't leaking between sessions.

## How to add phase rows below

When we start implementing Swift-side changes, each phase gets:
1. A new results file: `bench-results/tps-swift-phase<N>.txt`
2. A new cache-leak file: `bench-results/cache-leak-swift-phase<N>.txt`
3. A new column added to both tables above. Never overwrite old rows —
   the whole point of this file is to see every step of the ladder.
