# Comparison — `mlx-community/Qwen3.6-35B-A3B-4bit` (uniform-ish 4-bit INT)

Separate file from `COMPARISON.md` because the model differs: the original
file compares runs on the Unsloth UD-MLX-4bit model. This one is on the
mlx-community 4bit build (uniform 4-bit base with `mlp.gate` and
`shared_expert_gate` at 8-bit).

Hardware/env unchanged — see `ENVIRONMENT.md`.

## Decode TPS (higher is better — user-perceived streaming speed)

| scenario                 | ct   | Python 0.31.2 | Swift phase 1b (dev) | **Swift phase 2 (dev-moe + quant fix)** | Δ vs phase 1b |
|--------------------------|-----:|--------------:|---------------------:|----------------------------------------:|--------------:|
| short prompt, long gen   |  400 |          82.9 |                 62.9 |                                    88.4 | **+40%**      |
| long prompt, short gen   |   60 |          79.4 |                 73.0 |                                    76.3 | +4%           |
| opencode-like (~30k pt)  |   60 |          79.4 |                 45.9 |                                    47.9 | +4%           |
| multiturn turn 1         |   50 |          82.9 |                 39.1 |                                    90.0 | **+131%**     |
| multiturn turn 2         |   50 |          83.0 |                 39.0 |                                    89.8 | **+130%**     |
| multiturn turn 5         |   50 |          82.8 |                 39.8 |                                    89.7 | **+125%**     |

**Headline.** tom-eric's merged MoE/GDN kernel work (fused decode GDN,
fused gate+up MoE projection, async KV eval, etc.) more than **doubles**
multiturn decode TPS vs dev's PR #62 server, and **beats the Python
reference** (90 vs 83 tps on the exact workload opencode generates).
Long-context (4k, 30k) decode sees only a modest +4% bump — those scenarios
aren't MoE-routing-bound so the fused-MoE commits don't help there.

## Prefill TPS (higher is better — affects TTFT)

| scenario               | pt    | Python 0.31.2    | Swift phase 1b   | Swift phase 2     |
|------------------------|------:|-----------------:|-----------------:|------------------:|
| short prompt           |    36 | 174 (first 33)   | 192 (stable)     | 134 (stable)      |
| long prompt 4k         |  4032 | —                | 574              | 176               |
| opencode-like 30k      | 30028 | —                | 168              | 125               |
| multiturn turn 5       |   115 | 526              | 127              | 176               |

Caveat: prefill on dev-moe looks lower than dev here but *varies little*
between repeated runs on the same prompt. It's genuinely cold every run —
dev's higher numbers on runs 2+ were likely the server's session-cache LCP
hit from a lingering prior session. Comparing **first-run cold** numbers
(phase 1b "first=X" values in COMPARISON.md) is the fair read; those show
dev-moe and dev ~on-par for cold prefill. We'd need a cold-only harness to
be definitive about prefill; for decode the story is clean.

## Cache / leak checks (bench-results/cache-leak-*-mlxc-*.txt)

| check                              | python | swift phase 1b | **swift phase 2** |
|------------------------------------|:------:|:--------------:|:-----------------:|
| same-session recall (3 words)      |   ✅   |       ✅       |         ✅        |
| same-session follow-up names COMET |   ✅   |       ✅       |         ✅        |
| no KV leak across sessions         |   ✅   |       ✅       |         ✅        |
| new-conv follow-up still works     |   ✅   |       ✅       |         ✅        |
| bulk-prompt cache reuse (cold→warm)|   ✅   |       ✅       |         ✅        |
| cold TTFT on ~1500-token prompt    |  2.12s |      11.86s    |       9.34s       |
| warm TTFT (same prompt, turn 2)    |  0.58s |       0.94s    |       0.84s       |
| warm `cached_tokens` reported      |   n/a  |       1593     |       n/a¹        |

¹ Phase 2 (single-file server from tom-eric) doesn't emit
`prompt_tokens_details.cached_tokens` — only dev's PR #62 server does. Not a
correctness problem, just observability. Its TTFT drop (9.34s → 0.84s) is
consistent with a real cache hit.

## What this means for the plan

The next phase of work is to **merge dev-moe's wins back into dev** so we
keep PR #62's server (Anthropic API, sequential fix, usage accounting) AND
tom-eric's decode perf. Expected conflicts: Qwen35.swift (GDN fused decode),
SwitchLayers.swift (fused MoE projection). Bigger chunk of merge work than
the quant-fix port, but probably the right final step.
