# Retrieval Sweep — 2026-06-24

Canonical snapshot for the toolbox sweep Greg asked for. The point: **lay every hammer on the table, split P vs NP, sweep hyperparameters per tool, one script.**

## What's in here

```
src/retrieval/         — Greg's existing retrieval framework (June 15)
                         Feature × Scorer × Cue × Layer. Cue is a STRING.
                         base.py / features.py / scorers.py / pipeline.py
                         experiments/ has the holdout + real-corpus drivers.
data/uncertainty_dataset.py
                       — GENUINE_UNCERTAINTY (20) vs PERFORMED_CONFIDENCE (20).
                         Already wired in src/retrieval/datasets.py.
data/grief_originals.json    — Greg's 8 POS / 8 NEG grief, from exp_synthetic_probe.json.
data/flirt_originals.json    — Greg's 4 FLIRT / 4 NEUTRAL, from exp_reverse_attention_adjoint_flirt.json.
                                  ("Catch you at the meeting next Tuesday" lives here.)
data/stimulus_battery_20260623, stimulus_battery_v4
                       — Scaled latent-pragmatics cohorts (flirt/grief/apology/sarcasm/deception/uncertainty
                         × EN/ES/ZH), from the June 23 night. Useful for the cross-language /
                         scaled-n sweep AFTER the canonical-cue sweep on originals.
data/audit_v3, audit_v4
                       — TF-IDF leakage audits per cell. Marginal-token analysis.
                         Useful as "is the stimulus separable by surface alone" filter.

baselines/p_class_supervised_runner.py
                       — Yesterday's unified sweep (TFIDF logistic + linprobe + SBERT-hyde +
                         QK-perhead with mean-of-K-exemplars). P-class / K-shot-P-tinted.
                         Reference; superseded by the canonical script below.
baselines/p_class_perchunk_runner.py
                       — LOO per-chunk version of the above on Greg's originals.
baselines/exp_concept_cue.py, exp_real_corpus.py, exp_q_object_grid.py, exp_unbiased_sweep.py
                       — Earlier Greg-authored retrieval experiments. Already-frozen specs.

docs/SYNTHESIS_RAG_audit_v3.md
                       — Per-cell results table from yesterday's sweep with TF-IDF + SBERT
                         + supervised QK. Final headline + caveats.
```

## The frozen ship spec (from `src/retrieval/experiments/04_real_corpus.py`, June 15)

> Chunk feature: **Q(h)** at every token
> Cue feature: **Q(h)** at every token
> Scorer: **cross_cos_mean** (mean cosine across all (i,j) position pairs)
> Layer: **L27** on Qwen2.5-3B

This is the production NP-pure retrieval method. **It takes a string cue, encodes it once, dot-products against precomputed corpus chunks. No labels.**

## The two toolboxes Greg wants laid out

### NP-class (zero corpus labels, one query string)

Each (Feature, Scorer, Layer) is one tool config. Hyperparameters are the axes of `Feature` (k for innovation depth, base for projections) and `Scorer` (k for TopK).

| Tool | Feature | Scorer | Hyperparameter axes |
|---|---|---|---|
| q_object_grid | `QProj(HiddenState)` | `cross_cos_mean` | layer |
| h_raw_cos | `HiddenState` | `CosineLast` | layer |
| h_raw_cross_max | `HiddenState` | `CrossMax` | layer |
| h_raw_cross_softmax | `HiddenState` | `CrossSoftMax` | layer |
| h_raw_topk | `HiddenState` | `CrossTopK(k)` | layer, k ∈ {5, 20} |
| k_proj_cos | `KProj(HiddenState)` | `CosineLast` | layer |
| v_proj_cos | `VProj(HiddenState)` | `CosineLast` | layer |
| delta_cos | `Delta` | `CosineLast` | layer |
| delta_cross_max | `Delta` | `CrossMax` | layer |
| seq_innovation | `SequenceInnovation(k)` | `CosineLast` | layer, k ∈ {2,4,8} |
| seq_innovation_q | `QProj(SequenceInnovation(k))` | `cross_cos_mean` | layer, k |
| depth_innovation | `DepthInnovation(k)` | `CosineLast` | layer, k ∈ {2,4} |

PLUS the external NP baseline:
| Tool | Description |
|---|---|
| sbert_cue_cos | SBERT.encode(cue_string) · SBERT.encode(chunk_text). The "real RAG" Greg means. |

### P-class (supervised — fantasy world, useful as ceiling)

| Tool | Description |
|---|---|
| linprobe_resid | Ridge logistic on last-token resid per layer. 5-fold AUC. Ceiling. |
| linprobe_Q | Ridge on `QProj(HiddenState)` last-token. |
| linprobe_K | Ridge on `KProj(HiddenState)` last-token. |
| linprobe_pooled | Mean-pool resid across tokens. |
| tfidf_logistic | 1-2 gram TF-IDF + logistic. Pure surface. |

## The cues (NP-pure query strings) — to be wired

| Dataset | Cue string |
|---|---|
| uncertainty | "genuinely uncertain exploring possibilities conditional reasoning diagnostic hedging" (already wired) |
| grief | TBD — "a quote you would hear from a grieving person: my heart is torn, I miss them, the chair stays empty" |
| flirt | TBD — "subtle romantic interest, noticing details, hint of attraction without saying it" |
| apology_sincere | TBD — "a sincere apology naming what I did wrong and what I will do differently" |

## The script (next step, after this snapshot)

`retrieval_2026-06-24/THE_SWEEP.py` — single script that:

1. Loads each dataset (uncertainty, grief, flirt, [later: apology/sarcasm/deception/grief_es/flirt_zh/...])
2. For each dataset, runs `run_grid` over the full NP toolbox at layers {12,18,22,26,27,28,29,30,32,34,35} with the cue string.
3. Computes SBERT-cue NP baseline.
4. Computes the P-class probe ceiling per layer.
5. Writes one JSON per dataset with: per-config AUC, best NP config, SBERT baseline, P-class ceiling, NP-vs-RAG gap, NP-vs-P gap.
6. Writes one summary markdown with the toolbox table per dataset.

## What to look at first when the run finishes

For each dataset:
- Did any NP tool beat SBERT_cue? By how much? At what (feature, scorer, layer)?
- Where is the P-class ceiling? How close did NP get?
- Which tool is the consistent best across datasets? That's the production tool.
- Which dataset can NP NOT solve (NP ≤ SBERT, far below linprobe)? That's where the geometric claim breaks.
