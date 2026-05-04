# Trajectory Analysis Session — 2026-04-04

## Context
Per-token hidden state trajectories captured on **Qwen2.5-3B** (36 layers, d=2048) locally on RayGun.
16 problems across 4 domains (math, commonsense, code, logic), all 36 layers recorded, per-token cosine(ZH, EN) and norms saved.
Data: `output/trajectories_Qwen_Qwen2.5-3B_all.json` (6.7MB).

---

## Greg's Visual Analysis (verbatim observations)

### Cosine Heatmaps (Layer x Token, per problem)
- "Super clear horizontal drop at L12. Clear as day." Red ribbon (low cosine) on top layers (L0-L6), green (high cosine) on bottom layers (L24+). L12 is the visual boundary.
- Trivial arithmetic problems (P0-P2) are "trivially green" — high cosine everywhere below L6.
- "Let n be the smallest positive integer" (P3, CRT) looks very similar to the simple problems, just longer. Gets notably greener in early layers from around t=0.5 forward.
- Hard to evaluate apples-to-apples because of how t normalizes — different token counts mean different temporal resolution.

### Convergence by Domain (Mean cosine vs layer, the "money shot")
Greg described the full layer-by-layer story:

1. **L0→L1**: "Huge jump. Immediately some sort of Procrustes-kind of shift." Biggest single-layer gain for every domain. Math: 0.299→0.472. Code: 0.235→0.416.

2. **L1→L9**: Math and code start highest, then all domains decay. Math/code decay the most. "Everybody slows down and actually decays, but logic, code, and math tend to decay the most."

3. **L9**: Ranking starts to invert. Logic and commonsense "start to overtake, literally completely reversed." Before L9: math > code > logic > commonsense. After L9: logic > commonsense > math > code.

4. **L10-L12**: "Super tightly convergent." All domains come together. "They all kinda come to this choke point L12."

5. **L12→L25**: "They start to separate again" but move together. "Not co-integrated, but correlated — one guy goes up by X, other guy goes up by a square to X. You know what I'm saying? It's like they all move together in the same kind of way, just in different distances."

6. **L25-L27**: "Logic seems to have a drop at L25-26 that nobody else does, and math has a drop at L26-27 that nobody else does, whereas commonsense and code seem to just kind of be chilling and rallying."

7. **L30-L34**: "They all jump up together." Rally phase. L31-L34 final sprint.

8. **L34→L35**: "Huge drop off. That's a basin." The re-dressing crash.

9. Shadow color regions = confidence intervals (std bands). Correctly identified.

### Cosine Velocity
- Wanted L1, L9, L12, L26-27, L34 added to the velocity plot (currently only shows L0/L9/L18/L27/L35).
- "L12 would really benefit from being in this plot."

### Cosine Volatility
- "Huge peaks around t=0.6 across all layers for code, math, and logic."
- L18 and L27 more compressed.
- L0 much more hectic.
- Commonsense has a huge rally at t=0.1 at L0, whereas all other domains jump up at t=0.0.

---

## Numerically Confirmed Findings

### The L9-L12 Funnel
All four domains converge to within 1.5% spread at L9 and L12. This is a **universal representation bottleneck**.

| Layer | Spread (max-min) | Note |
|-------|-----------------|------|
| L0 | 0.1242 | Wide — math/code high, commonsense/logic low |
| L1 | 0.1185 | Still wide after L1 jump |
| L7 | 0.0355 | Starting to tighten |
| L8 | 0.0263 | Funnel |
| **L9** | **0.0151** | **Tightest point in entire stack** |
| L10 | 0.0287 | Funnel |
| L11 | 0.0369 | Tight |
| **L12** | **0.0153** | **Second tightest** |
| L13 | 0.0222 | Funnel |
| L15 | 0.0515 | Starting to diverge |
| L20 | 0.0947 | Diverged |
| L25 | 0.1235 | Max divergence zone |
| L34 | 0.0620 | Reconverging at peak |
| L35 | 0.1216 | Crash — re-dressing divergence |

### Domain Ranking Inversion
| Layer | Ranking (highest cosine to lowest) |
|-------|-----------------------------------|
| L0 | math > code > logic > commonsense |
| L1 | math > code > logic > commonsense |
| L9 | logic > math > code > commonsense |
| L12 | commonsense > math > logic > code |
| L25 | commonsense > logic > math > code |
| L34 | commonsense > logic > code > math |
| L35 | math > logic > commonsense > code (INVERTS at re-dressing) |

### L35 Re-dressing Paradox
Code crashes hardest at L35 (0.369) despite outputting Python in both languages. Math crashes least (0.490) despite outputting different-language text. Hypothesis: L35 crash measures scaffolding cost (comments, explanations around the code), not the code itself.

---

## Vega's Open Questions (2026-04-04)

1. **Does the L12 funnel exist at the same relative depth on larger models?** On 3B (36 layers) it's at L9-L12 = 25-33% depth. On 8B (36 layers, same arch) it should be same absolute layers. On 14B (48 layers) it would be L12-L16 if relative, or L9-L12 if absolute. This distinguishes architectural vs capacity effect.

2. **Why does the ranking invert?** Hypothesis: math/code share more surface tokens (numbers, operators, Python keywords) which inflate early-layer cosine. Commonsense/logic share more semantic content which only aligns in deeper layers.

3. **Are the L25-27 adversarial dips domain-specific?** Logic and math dip, commonsense doesn't. Does this mean the adversarial phase is task-dependent? Connects to Exp T adversarial/cooperative phase transition.

4. **Why does code crash hardest at L35?** Code output is Python in both languages — re-dressing cost should be LOWER. Unless L35 measures scaffolding (comments, docstrings) not code.

5. **What's at t=0.6?** The volatility spike Greg noticed. Does this correspond to the reasoning→answer transition? Need to align with actual token content to know.

---

## Next Experiments (priority order)

1. **8B trajectory capture** (Colab A100) — same architecture, direct comparison
2. **Qwen3.5-9B trajectory capture** — different architecture family, recent model. Greg: "You've been sleeping on that."
3. **Updated velocity plot** with L1, L12, L26, L34 (Greg's request)
4. **Phase-aligned analysis** — align by computation phase (problem restatement / reasoning / answer) instead of normalized time
5. **Retroactive trajectory plots** for earlier experiments (Exp T/U/V/W/X/Y/Z)

---

## Key Files
- `output/trajectories_Qwen_Qwen2.5-3B_all.json` — raw per-token data (6.7MB)
- `output/fig_cosine_heatmaps.png` — 16-panel layer x token heatmaps
- `output/fig_convergence_by_domain.png` — the money shot
- `output/fig_cosine_velocity.png` — d(cos)/dt at key layers
- `output/fig_cosine_volatility.png` — rolling realized vol
- `output/fig_norm_ratio.png` — ZH/EN norm ratio through layers
- `output/fig_layer_jumps.png` — layer-to-layer cosine delta heatmaps
- `output/fig_deep_dive.png` — full Dijkstra breakdown
- `output/fig_norm_vs_cosine_L18.png` — norm tracking + cosine at convergence zone
- `perturbation/exp_trajectory_capture.py` — trajectory recorder script
- `moams_x_analysis.ipynb` — unified notebook (14 sections, sections 8a-8h are trajectory)
