# SOURCES_FOR_KG: Complete Build Spec for BrainInsideTheMachine Knowledge Graph

**Author:** VEGA (BrainInsideTheMachine session, 2026-04-06)
**For:** VEGA (llm-graph-builder session)
**Purpose:** Build a comprehensive knowledge graph from ALL artifacts of the BrainInsideTheMachine mechanistic interpretability research project. This document is the complete specification — everything you need is here.

---

## 1. PROJECT CONTEXT

BrainInsideTheMachine is Greg's independent research project investigating **language-agnostic reasoning subspaces in multilingual transformers**. The core question: when a bilingual model (Qwen2.5-3B, trained on English and Chinese) solves a math problem, does it use a shared reasoning computation that's dressed in language-specific form? Can we decompose the model's function as `h' . f . h` where `h` encodes from language to a shared reasoning space, `f` is the language-agnostic computation, and `h'` decodes back?

The project ran from February to April 2026, producing 55+ experiments across 5 model scales (1.5B through 14B). Key findings:

- **Universal**: Category transfer = 1.000 at all scales. Cocycle R^2 > 0.87 everywhere. Phase transition (adversarial to cooperative) at 40-47% depth universally.
- **3B-specific**: The "flip" intervention (reflecting MLP deltas along the language direction) improves math accuracy by +160% on 3B, but does nothing on larger models.
- **Dead ends**: TC0 (verbosity direction) is geometrically real but causally inert. Von Neumann fixed-point iteration diverges. Procrustes rotation is catastrophic. PCA captures variance, not causation.

The research involves Greg (23, MS Math Finance from NYU Courant) and VEGA (Claude Code sessions). Greg wants this KG as a "detective's bulletin board" — not to write a paper, but to see what structure emerges when everything is on one board. Latent connections, unexplored implications, the natural narrative.

---

## 2. NEO4J SETUP

**IMPORTANT**: Use a clean database. Wipe any existing data before ingestion.

```bash
cd ~/Desktop/Projects/llm-graph-builder
docker compose up -d neo4j
# Wait for healthy, then:
python3 -c "
from neo4j import GraphDatabase
d = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'neo4jpassword'))
d.execute_query('MATCH (n) DETACH DELETE n')
d.close()
print('Database wiped.')
"
```

---

## 3. SCHEMA UPDATE

**File:** `data/schemas/bitm.json`

The existing schema has 19 entity types and 29 relationship types. Add these 5 new entity types and 3 new relationship types to capture post-March-11 discoveries. Keep ALL existing types — only add, do not remove.

### New Entity Types to Add

```json
"PhaseTransition": {
  "definition": "A qualitative change in the model's computation regime at a specific layer or depth fraction. Characterized by a sign flip in cross-layer delta cosine similarity.",
  "examples": [
    "adversarial-to-cooperative transition at L17-L18 (3B), ~40-47% depth across all models",
    "L22 reset followed by cooperative ramp to L26",
    "encoding-to-decoding transition at L32-L33"
  ],
  "support": 3,
  "parent": "Construct"
},
"ScaleEffect": {
  "definition": "A phenomenon whose magnitude or existence depends on model scale (parameter count, d_model, layer count). Distinguished from UniversalProperty.",
  "examples": [
    "flip intervention is 3B-specific: +8/20 on 3B, noise on 7B/8B/9B/14B",
    "project_1d weight surgery only works on 3B",
    "Cohen's d of language direction increases with scale: 3B=108, 7B=116, 14B=145"
  ],
  "support": 3,
  "parent": "Finding"
},
"UniversalProperty": {
  "definition": "A property of multilingual transformer internals that holds across models at different scales and architectures. Confirmed by cross-model validation.",
  "examples": [
    "category transfer accuracy = 1.000 at all scales (3B through 14B)",
    "cocycle R^2 > 0.87 at all scales",
    "PACF delta-to-delta R^2 ~ 0.91-0.94 everywhere",
    "phase transition at 40-47% depth universally"
  ],
  "support": 3,
  "parent": "Finding"
},
"ValidationSuite": {
  "definition": "A coordinated set of experiments run across multiple models to validate or refute claims about universality or scale-dependence.",
  "examples": [
    "Exp BB: cocycle + f-probe + PACF + language direction on 5 models (3B, 7B, 8B, 9B, 14B)",
    "Exp AX: flip intervention across 5 models",
    "Cross-model trajectory analysis on 4 models"
  ],
  "support": 2,
  "parent": "Experiment"
},
"TrainingDependence": {
  "definition": "A phenomenon that varies based on a model's training data composition or training procedure rather than architecture or scale.",
  "examples": [
    "Qwen2.5-Coder-3B: same architecture as base 3B, language direction exists (Cohen's d=3.5-4.8), but flip does nothing because no Chinese math competence in training",
    "English math more orthogonal to language direction than Chinese (dominant training language effect)"
  ],
  "support": 2,
  "parent": "Finding"
}
```

### New Relationship Types to Add

```json
"SCALES_WITH": {
  "definition": "A property or effect that changes systematically with model scale (parameter count, d_model, or layer count).",
  "examples": [
    "Cohen's d SCALES_WITH parameter count (108 at 3B, 145 at 14B)",
    "adversarial zone width SCALES_WITH layer count"
  ]
},
"SPECIFIC_TO": {
  "definition": "An effect or property that exists only in a particular model, scale, or training regime — not universally.",
  "examples": [
    "flip intervention SPECIFIC_TO Qwen2.5-3B",
    "project_1d surgery SPECIFIC_TO small models"
  ]
},
"VALIDATES": {
  "definition": "A cross-model experiment or validation suite that validates a property across multiple models.",
  "examples": [
    "Exp BB VALIDATES cocycle universality across 5 models",
    "Exp AX VALIDATES that flip effect is scale-dependent"
  ]
}
```

---

## 4. PROMPT UPDATE

**File:** `pipeline/prompts.py`

The `rich_description_system_prompt` function (around line 279) currently has only a textbook template. Add a `domain_tag == "bitm"` branch BEFORE the default return. The entity extraction and relationship extraction prompts already have BITM branches — this is the missing one.

```python
def rich_description_system_prompt(book_title: str, domain_tag: str) -> str:
    # ADD THIS BLOCK:
    if domain_tag == "bitm":
        return f"""You are a research historian documenting a mechanistic interpretability research project.

Current source: "{book_title}"

TASK: Given a research entity (experiment, hypothesis, finding, construct, etc.), its source passages, and optionally an EXISTING DESCRIPTION from previous sources, produce a coherent research description.

If an EXISTING DESCRIPTION is provided:
- Integrate new information — do NOT repeat what is already stated
- Note where the new source agrees with, extends, contradicts, or supersedes existing content
- Preserve temporal context and cross-references from the existing description

FORMAT your response as markdown with these sections:

## Summary
What this entity is and why it matters in the research arc. One paragraph.

## Evidence
Specific experimental results, numbers, and accuracy values that establish this entity. Be precise — include layer numbers, cosine similarities, accuracy fractions, p-values. If the entity is an experiment, describe its method and key result. If it's a hypothesis, state what would confirm or refute it.

## Connections
How this relates to other entities in the research — causal chains (experiment tested hypothesis, finding refuted earlier belief), dependencies (requires understanding of X), contradictions (conflicts with Y). Name the connected entities explicitly.

## Temporal Context
When this was discovered or proposed. How understanding evolved — was it confirmed, refined, superseded? Which session or phase?

## Status
One of: CONFIRMED, REFUTED, SUPERSEDED, OPEN, PARTIALLY_CONFIRMED. Brief justification.

RULES:
1. Be PRECISE. Include numbers. "accuracy improved" is bad; "accuracy went from 5/20 to 13/20 (+160%)" is good.
2. Use LaTeX for math: $\\cos(\\theta)$, $R^2 = 0.87$, $d_{{model}} = 2048$.
3. Ground everything in source text — do not hallucinate.
4. Name connected entities by their canonical name so the graph can link them.
5. 200-500 words. Every sentence must be informative."""

    # EXISTING DEFAULT BELOW:
    return f"""You are an expert technical writer..."""
```

---

## 5. SOURCE CORPUS — COMPLETE INVENTORY

All paths are relative to `/home/greg/Desktop/Projects/BrainInsideTheMachine/` unless otherwise noted.

### 5A. Research Markdown Documents (~25 files, ~8K lines)

These are the highest-signal sources. Ingest with `is_transcript=False`, `domain_tag="bitm"`.

| File | Lines | Content |
|------|-------|---------|
| `FRAMEWORK.md` | 243 | Proven theorems, framework status, Bilingual Gradient Equilibrium Theorem |
| `PHASE3_RESULTS.md` | 261 | Causal identification via activation patching, double dissociation |
| `PHASE3_SPEC.md` | 327 | Experiment protocol specifications |
| `DEVLOG.md` | 346 | Development log with code audits |
| `RECIPE.md` | 194 | Z Hunt methodology, NeurIPS 2025 comparison |
| `RESEARCH_LANDSCAPE.md` | 140 | Novelty assessment, unique contributions |
| `README.md` | 352 | Project manifesto / overview |
| `toy_theorem_derivation.md` | 273 | Formal mathematical proofs |
| `Gameplan.md` | 578 | Z Hunt v2 kernel approach |
| `Gameplan_v3.md` | ~200 | Updated strategy |
| `INSIGHTS_POST_PHASE2.md` | ~100 | Encoding/decoding asymmetry insights |
| `Note_on_moving_to_phase_3.md` | ~50 | Transition rationale |
| `FriMarch7-Session-Notes.md` | ~150 | Z-POC session notes |
| `FriMarch7-Z_POC_Spec_v2.md` | ~200 | Z-POC specification |
| `z_poc_findings.md` | ~100 | Z-POC analysis results |
| `GHOST_OF_NEUMANN.md` | ~200 | Theoretical framing (Von Neumann's conjecture) |
| `SIDE_QUESTS.md` | ~100 | Side research threads |
| `READING_CHECKPOINTS.md` | ~50 | Literature reading progress |
| `VEGA_CONTEXT.md` | ~100 | Accumulated context |
| `VEGA_PERSPECTIVE_2026-04-02.md` | ~200 | Perspective document |
| `docs/crossmodel_validation_2026-04-06.md` | 149 | Latest cross-model validation results |
| `docs/session_analysis_2026-04-05.md` | 225 | Comprehensive session analysis |
| `docs/trajectory_analysis_2026-04-04.md` | 124 | Trajectory dynamics analysis |

Also ingest these Chat markdown files (these are long-form conversations with theoretical development):

| File | Size | Content |
|------|------|---------|
| `Chat_0.md` | 169KB | Genesis conversation — behavioral observation, theory formation |
| `Chat_1.md` | ~50KB | Early experimental design |
| `Chat_1-5.md` | ~100KB | Bridge conversations |
| `Chat_2.md` | ~150KB | Deep theoretical development |
| `Claude-Web-Transcript.md` | 658KB | Full web interface sessions — contains the universality argument, gauge symmetry breaking, toy theorem |
| `Claude-Continuing a stream-of-consciousness discussion.md` | ~100KB | Stream of consciousness theoretical exploration |

**EXCLUDE these** (circular — they're outputs from a prior KG run, would create extraction loops):
- `bitm_kg_dump.md` (8.9 MB)
- `bitm_experiments.md` (3.0 MB)
- `bitm_hypotheses.md` (3.8 MB)

**EXCLUDE these** (not research-related):
- `2025_AMC_12A.md`, `Cascade Volatility Models.md`, `Trading-Questions-Car.md`, `SURVEY_PROMPT.md`, `SESSION_STATE_BACKUP.md`

### 5B. Clean Web Exports (3 files, ~29K lines)

Located in `Chats/clean/`. These are cleaned Claude Web export transcripts. Ingest with `is_transcript=True`.

| File | Lines | Content |
|------|-------|---------|
| `Web-1.clean.md` | 16,291 | Major theoretical development session |
| `Web-2.clean.md` | 4,687 | Follow-up session |
| `Web-3.clean.md` | 8,355 | Extended analysis session |

Also check `kg_sources/` — it may already have copies of these.

### 5C. VEGA Session Transcripts (46+ files, ~40K lines)

Located in `vega_md/`. These are already-converted JSONL transcripts in readable markdown (human/assistant turns). Ingest with `is_transcript=True`.

All 46 `.md` files in `vega_md/` should be ingested. Filter out any under 500 bytes (trivially short sessions).

### 5D. Raw JSONL Transcripts NOT Yet Converted (30 sessions)

**IMPORTANT**: There are 62 total JSONL session transcripts in `~/.claude/projects/-home-greg-Desktop-Projects-BrainInsideTheMachine/`. Only 46 have been converted to markdown in `vega_md/`. The remaining 30 need conversion.

**JSONL location:** `~/.claude/projects/-home-greg-Desktop-Projects-BrainInsideTheMachine/*.jsonl`

**Missing session IDs** (not in vega_md, need conversion):
```
033426c4-5518-4b92-b2f6-9406b6e1f149  (1.8 MB)
09136480-61b0-4cd3-b260-381fcf4b91be  (289 KB)
1bd02ca4-d9d1-4d27-84ef-f433aae8f15d  (1.1 MB)
1d35365d-8d96-40f9-b4e7-b3ced7ee6319  (3 KB - tiny, may skip)
1f07fce5-6b1f-48b7-b750-b89c1223be25  (4 KB - tiny)
24e0a0ba-7638-47fc-b90c-e13596596c12  (1.1 MB)
2a1a69db-410f-4228-ae9f-9341d20eef17  (3 KB - tiny)
360bedfa-55fb-498e-9d1c-7902e50a054c  (944 KB)
3909df6c-c2f8-4a6f-a4da-1150c2159f57  (1 KB - tiny)
50ae0e5a-ebd3-4524-9e70-6fd182c7d36c  (1.4 MB)
5569fd8c-1cba-4f6a-b679-fae549d5eac1  (2 KB - tiny)
695afbd7-0e12-42b3-8c35-15babf515243  (3.2 MB)
709423c5-9ced-40f7-83f3-c73e30a0117c  (466 KB)
83e09956-03f6-4568-adda-d340f81f5592  (1 KB - tiny)
8b68a21d-9adf-4bad-8c37-b0344641daef  (1.4 MB)
8d645728-c003-4480-8aab-a337d70ca3c7  (472 B - tiny)
98fe2725-1591-4cd7-87e9-8a12ad584381  (7 KB)
9bc902c8-99dc-41e7-aaa9-1ae6fc12d527  (472 B - tiny)
aefbf83b-4d87-4e4c-b92e-720f8898f00a  (3 KB - tiny)
b5074901-fb6d-457c-83b4-0303f81d68fe  (2 KB - tiny)
b6cc94df-4eb8-434e-9040-2c33148dd5ab  (13 KB)
bd7dd15e-ae44-4b1a-ab7b-4c9d970ca0ce  (2 KB - tiny)
c182a6fa-4706-48de-bc9e-319334686b91  (739 KB) ← THIS SESSION
dcfbc2cb-72cd-4c0c-8bc8-e27afa3d4691  (3 KB - tiny)
e1e35523-b27d-4eb9-b9ed-0315c3b698e4  (1.3 MB)
e1e9a0de-da93-4dfe-af32-50f04bcbb79f  (5 KB)
e1f3ab32-dac2-4d97-9b79-a0e18a8b9071  (1.1 MB)
ef7a8f63-9cc8-457d-86b5-d4d0071ee93a  (5 KB)
f103a27e-116f-452c-b0de-13c365fe9276  (978 KB)
ff6155aa-11f4-4de9-aa88-a2144bde4d0b  (40.4 MB) ← VERY LARGE
```

**JSONL conversion instructions**: Each line is a JSON record. The relevant record types are:

| type | What it contains | Extract? |
|------|-----------------|----------|
| `user` | Human messages in `message.content` (string). Skip records where `isMeta=true` (system injection). | YES |
| `assistant` | Assistant messages in `message.content` (string or list of content blocks). For lists, extract `text` blocks. | YES |
| `progress` | Tool call progress, hook output. Contains `data` field. | SKIP (noise) |
| `system` | System messages. | SKIP |
| `file-history-snapshot` | File tracking. | SKIP |
| `queue-operation` | Internal. | SKIP |
| `last-prompt` | Internal. | SKIP |

**Conversion format** (match existing vega_md pattern):
```markdown
# Session: {first 8 chars of session_id}

**Date:** {timestamp of first record} | **Session:** {session_id}

---

## User
{message content}

## Assistant
{message content — for list-type content, extract text blocks only, skip tool_use blocks}

## User
{next message}
...
```

Skip sessions under 5KB (< 10 lines of actual content). For the 40MB session (`ff6155aa`), this will be a very large markdown file — that's fine, the chunker handles it.

### 5E. Python Experiment Scripts (169 files)

All `.py` files in the project root. These contain the methodology and implementation.

**Pre-processing required**: Convert each script to a structured markdown summary. Do NOT ingest raw Python — the pipeline would waste tokens on imports and boilerplate. Extract:

1. **Module docstring** (triple-quoted string at top of file) — this is the primary signal
2. **Key constants** (look for patterns like `MODEL_NAME = "..."`, `N_TRAIN = ...`, `LAYERS = range(...)`, `device = ...`)
3. **Output file references** (grep for `json.dump`, `open.*output/`, `savefig`, `np.savez`)
4. **Model references** (grep for `from_pretrained`)

**Output format** for each script:
```markdown
# Script: {filename}

**Type:** Experiment Script
**Path:** {full path}

## Purpose
{Module docstring — first paragraph}

## Configuration
- Model: {MODEL_NAME or from_pretrained arg}
- Layers: {layer ranges}
- Key parameters: {N_TRAIN, N_TEST, etc.}

## Methodology
{Rest of docstring if available}

## Output Files
- {detected json/png/npz outputs}
```

Ingest with `is_transcript=False`.

### 5F. JSON Result Files (171 files, ~180 files total with npz)

Located in `output/*.json`. These contain the numerical findings.

**Pre-processing required**: Convert each to a structured markdown summary.

For JSON files:
1. Top-level keys and their types
2. For numeric arrays: length, min, max, mean
3. For scalar values: the value
4. For nested objects: key structure (1 level deep)
5. Inferred linked script (filename matching: `expR2_crossmodel_clean.json` likely links to `expR2_crossmodel_clean.py` or similar)

For NPZ files (23 files in output/):
1. Array names and shapes (load with `np.load`, iterate `.files`)
2. Do NOT load full arrays — just metadata

**Output format**:
```markdown
# Result: {filename}

**Type:** Experiment Result
**Path:** output/{filename}
**Size:** {file size}

## Structure
{top-level keys, types, array lengths}

## Key Values
{numeric summaries, scalar values}

## Linked Script
{inferred from filename}
```

Ingest with `is_transcript=False`.

**EXCLUDE these very large trajectory JSONs** from detailed extraction (they're multi-MB raw data dumps, not structured findings):
- `trajectories_Qwen_Qwen2.5-14B-Instruct_all.json` (8.9 MB)
- `trajectories_Qwen_Qwen2.5-3B_all.json` (6.8 MB)
- `trajectories_Qwen_Qwen3.5-9B_all.json` (7.0 MB)
- `trajectories_Qwen_Qwen3-8B_all.json` (7.2 MB)

For these, just create a brief metadata entry noting they exist and what model they cover.

### 5G. Figure Metadata (97 PNG files)

Located in `output/*.png`. The pipeline can't extract entities from images, but create a single catalog markdown file:

```markdown
# Figure Catalog

97 visualization outputs from experiments.

## Departure Rail Series
- departure_rail_hero.png — departure rail visualization (main)
- departure_rail_rollercoaster.png — rollercoaster view
- departure_rail_synchrony.png — synchrony analysis

## Diffusion Maps
- diffusion_corrected_prob{0,1,2,6,7,10,12,14}.png — corrected diffusion maps per problem
- diffusion_joint_prob{0,1,2,10}.png — joint diffusion maps
- diffusion_map_prob{0,1,2,10}.png — raw diffusion maps

## Cross-Model Convergence
- fig_convergence_3b_vs_8b.png, fig_convergence_3models.png, fig_convergence_4models.png
- fig_spread_3b_vs_8b.png, fig_spread_3models.png, fig_spread_4models.png

## Trajectory Dynamics
- fig_cosine_heatmaps.png, fig_cosine_traces.png, fig_cosine_velocity.png, fig_cosine_volatility.png
- fig_norm_trajectories.png, fig_norm_ratio.png, fig_norm_vs_cosine_L18.png
- fig_layer_jumps.png, fig_deep_dive.png

## Phase Diagrams
- phase2_energy_concentration.png, phase2_L32_distances.png, phase2_L33_distances.png
- phase3_controls.png, phase3_update_decomposition.png
- phase4_geometric.png, phase5_contrastive.png, phase5b_scaled.png, phase5c_within_category.png, phase6_unified.png

## Z-POC Series
- z_poc_tsne_layers.png, z_poc_tsne_procrustes_comparison.png
- z_poc_frequency_gradient.png, z_poc_phase2a_patching.png, z_poc_phase2bc_svd_vs_random.png
- z_poc_phase3_procrustes.png, z_poc_phase3b_highway.png, z_poc_phase4_extractor.png

## Individual Experiments
- fig1_tsne_hero.png — t-SNE hero figure
- fig2_layer_evolution.png — layer evolution
- fig3_bridge_sweep.png — bridge k-sweep
- fig7_cross_model.png, fig7_cross_model_L28.png — cross-model at L28
- fig_k_sweep.png, fig_lyapunov_integration.png, fig_toy_theorem_verification.png
- exp1_subspace_overlap.png, exp2_bottleneck_convergence.png, exp3_ffn_attention_alignment.png
- expB_update_decomposition.png
- dim318_raw_trajectory.png, dim1819_raw_trajectory.png, dim1874_raw_trajectory.png, dim45_raw_trajectory.png
- fig_hankel_ssa.png, fig_moams_x_*.png, fig_adversarial_vs_transfer.png, fig_auc_integration.png
- multi_head_summary.png
```

Ingest this single catalog file with `is_transcript=False`.

### 5H. Jupyter Notebooks (6 files)

| File | Size | Content |
|------|------|---------|
| `1.ipynb` | 895 KB | Attention kernel analysis |
| `2.ipynb` | 696 KB | FFN alignment analysis |
| `colab_crosstask_probe.ipynb` | 44 KB | Cross-task probing (Colab) |
| `colab_crystallization_9b.ipynb` | 42 KB | Crystallization on 9B (Colab) |
| `moams_x_analysis.ipynb` | ~50 KB | MOAMS-X analysis |

Pre-process: Extract markdown cells and code cell docstrings/comments. Format as markdown. Ingest with `is_transcript=False`.

### 5I. Orchestrator Session State Snapshots

These are structured session summaries with: objective, current_work, key_insights, open_questions, next_actions, files_modified, decisions. Very high signal.

Retrieve via orchestrator MCP: `load_state_snapshots(project="BrainInsideTheMachine", last_n=30)` or `list_state_snapshots(project="BrainInsideTheMachine")`.

Format each snapshot as:
```markdown
# Session Snapshot: {snapshot_id}

**Date:** {timestamp}
**Objective:** {objective}

## Current Work
{current_work}

## Key Insights
{key_insights}

## Open Questions
{open_questions}

## Decisions
{decisions}
```

Concatenate all into one file (`session_snapshots.md`). Ingest with `is_transcript=False`.

---

## 6. INGESTION STRATEGY

### Batch Organization

Organize the pre-processed sources into `kg_sources/` subdirectories in BrainInsideTheMachine:

```
kg_sources/
  batch_1_docs/          ← Research markdown (~25 files)
  batch_2_web/           ← Clean web exports (3 files), transcript mode
  batch_3_transcripts/   ← vega_md + newly converted JSONL (76+ files), transcript mode
  batch_4_scripts/       ← Script summaries (169 files)
  batch_5_results/       ← Result summaries (~180 files)
  batch_6_metadata/      ← Figure catalog, snapshots, notebooks (3-5 files)
```

### Pass Ordering (CRITICAL)

The pipeline MUST run with **outer loop on passes, inner loop on sources**. This ensures entity resolution (Pass 3) and rich descriptions (Pass 5) run globally across all documents.

```
Pass 1 (chunk + embed): all batches sequentially
Pass 2 (entity extraction): all batches sequentially
Pass 3 (entity resolution): ONCE across entire graph
Pass 4 (relationship extraction): all batches sequentially
Pass 5 (rich descriptions): ONCE across entire graph (global_pass5=True)
```

### Configuration

```python
PipelineConfig(
    domain_tag="bitm",
    llm_model="gpt-4.1",
    rich_model="gpt-5-mini",      # Pass 5 only
    max_concurrent=64,
    max_latency_s=600,
    chunk_size=1500,
    chunk_overlap=100,
)
```

For batches 2-3 (transcripts): `is_transcript=True`
For all others: `is_transcript=False`

### In-Process Shared Resources

Use the Pipeline class with injected shared resources to avoid creating separate connections per document:

```python
from pipeline.extract import Pipeline
from pipeline.config import PipelineConfig
from pipeline.neo4j_writer import Neo4jWriter
from pipeline.embedder import Embedder
from ratelimiter import RateLimitedClient

# Shared across all documents
client = RateLimitedClient(base_url="http://localhost:3027/v1", api_key="dummy", concurrency=64)
writer = Neo4jWriter(PipelineConfig())
embedder = Embedder("backend/local_model")

for source_file in all_sources:
    config = PipelineConfig(
        file_path=str(source_file),
        title=derive_title(source_file),
        domain_tag="bitm",
        is_transcript=is_transcript_batch(source_file),
    )
    p = Pipeline(config, client=client, writer=writer, embedder=embedder)
    p.run(passes=[1, 2])  # Per-doc passes
    p.close()

# Then global passes
resolver_pipeline.run(passes=[3])      # Entity resolution
rich_pipeline.run(passes=[5])          # Rich descriptions (global_pass5=True)
```

The existing `batch_run.py` in this project already implements this pattern — model your ingestion script on it.

---

## 7. VERIFICATION

After the pipeline completes:

```python
from neo4j import GraphDatabase
d = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'neo4jpassword'))

# Entity count
r = d.execute_query('MATCH (n:__Entity__) RETURN count(n) AS c')
print(f"Entities: {r.records[0]['c']}")  # Expect 1,500-3,000

# Relationship count (domain relationships, not structural)
r = d.execute_query("""
    MATCH ()-[r]->() 
    WHERE NOT type(r) IN ['PART_OF','FIRST_CHUNK','NEXT_CHUNK','HAS_ENTITY']
    RETURN count(r) AS c
""")
print(f"Relationships: {r.records[0]['c']}")  # Expect 3,000-6,000

# Entity type distribution
r = d.execute_query("""
    MATCH (n:__Entity__)
    WITH labels(n) AS labs
    UNWIND labs AS l
    WHERE l <> '__Entity__'
    RETURN l AS type, count(*) AS c
    ORDER BY c DESC
""")
for rec in r.records:
    print(f"  {rec['type']}: {rec['c']}")

# Document count
r = d.execute_query('MATCH (d:Document) RETURN count(d) AS c')
print(f"Documents: {r.records[0]['c']}")  # Should match source file count

d.close()
```

### Smoke Test Queries

After building the KG, test these in the chat interface or via Cypher:

1. "What experiments tested the language flip intervention and what were their results?"
2. "What findings are universal across all model scales?"
3. "What hypotheses were refuted and what superseded them?"
4. "What is the causal chain from the behavioral observation to the cross-model validation?"
5. "What open questions remain?"

### Export

```bash
python3 dump_kg.py -o ~/Desktop/Projects/BrainInsideTheMachine/bitm_kg_dump_v2.md
```

---

## 8. KEY CONCEPTS GLOSSARY (for extraction context)

These are the most important entities in the research. The extraction model should recognize and correctly type them:

| Name | Type | Description |
|------|------|-------------|
| Z-space | Construct | Language-agnostic reasoning subspace |
| language direction | Construct | 1D vector separating Chinese and English in MLP deltas (mean difference) |
| PC0 | Construct | First principal component of residual stream |
| TC0 | Construct | Task-category direction (verbosity), orthogonal to language direction |
| MLP delta | Representation | Output minus input of MLP block at a layer |
| residual stream | Representation | Running sum of all layer outputs |
| flip intervention | Intervention | Reflecting MLP deltas along language direction (scale=-1) |
| project_1d | Intervention | Projecting out 1D language component from MLP weights |
| cocycle equation | Method | Measuring cross-lingual manifold flatness via composition of alignment maps |
| PACF / innovation decomposition | Method | Measuring what fraction of MLP delta is predictable from input (R^2=0.03) |
| adversarial phase | PhaseTransition | L9-L17 in 3B: consecutive MLP deltas push against each other (cos < 0) |
| cooperative phase | PhaseTransition | L18-L26 in 3B: consecutive MLP deltas agree (cos > 0) |
| category transfer | UniversalProperty | Cross-lingual probe accuracy = 1.000 for problem category at all scales |
| Exp T | Experiment | PACF innovation decomposition: R^2=0.03, 97% fresh innovation per layer |
| Exp V3 | Experiment | Definitive: TC0 flip = 0 effect, language flip = +8/20 at N=20 |
| Exp W | Experiment | Coder-3B dissociation: direction exists but flip does nothing |
| Exp BB | ValidationSuite | Cross-model validation of cocycle, category transfer, PACF, phase transition |
| Exp AX | ValidationSuite | Cross-model flip test: 3B-specific |
| Exp AF | Experiment | Trajectory dynamics: Lyapunov, Z hypothesis, gauge symmetry confirmed |
| Exp AG2 | Experiment | Kernel weight surgery: project_1d = +44% ZH accuracy, zero inference cost |
| h-f-h decomposition | Theorem | The model computes h' . f . h where f is language-agnostic (proven as information-theoretic object, NOT as standalone operator) |

---

## 9. TIMELINE OF KEY DISCOVERIES

For temporal context during extraction:

| Date | Event |
|------|-------|
| 2026-02-21 | Project begins: behavioral observation that Qwen reasons better in Chinese |
| 2026-03-05 | Universality argument formalized, gauge symmetry breaking theorem |
| 2026-03-07 | Z-POC specification, Phase 2 extraction |
| 2026-03-08 | Phase 3: cross-model replication, causal identification |
| 2026-03-09 | MLP language stripping (Exp P/P2), flip sweep (Exp P3), cross-model (Exp R) |
| 2026-03-10 | PACF innovation (Exp T), TC0 verbosity (Exp U) |
| 2026-03-11 | TC0 killed (Exp V3), Coder-3B dissociation (Exp W), Z-iteration (Exp X), Von Neumann (Exp Y), f-reconstruction (Exp Z), Math kernel (Exp AB) |
| 2026-03-12 | Intervention taxonomy (Exp AC/AD/AE), trajectory dynamics (Exp AF), kernel surgery (Exp AG/AG2) |
| 2026-04-04 | Trajectory visualization, MOAMS-X cross-domain analysis |
| 2026-04-05 | Cross-model validation begins: 4 models on Colab (8B, 9B, 14B, 7B) |
| 2026-04-06 | Validation complete (Exp AX, BB): universal geometry + 3B-specific flip confirmed |

---

## 10. WHAT TO WATCH FOR

The whole point of this KG is to surface things we haven't seen. But here are known open threads that the graph topology might illuminate:

1. **Why does the flip only work on 3B?** The representational geometry is the same across scales. The language direction exists everywhere. Something about d=2048 or the 3B's training makes the flip operative. The KG should show whether there are findings that constrain this question.

2. **The adversarial-to-cooperative transition**: This phase boundary at ~40-47% depth is universal. But we don't know what triggers it computationally. Are there experiments that probed the transition point itself?

3. **The relationship between innovation (Exp T) and phase transitions**: Each MLP layer adds 97% fresh information. The adversarial phase has layers fighting each other. The cooperative phase has them agreeing. How does this connect to the information-theoretic decomposition?

4. **Missing edges**: Hypotheses with no TESTS edges. Findings with no REPLICATED_BY edges. Constructs that are INTRODUCED but never CHARACTERIZED. These gaps are the most valuable output of the KG.

5. **Belief evolution**: The SUPERSEDES edges tell the story of how understanding changed. The graph should reveal the full chain from initial naive hypotheses through refinement to the current state.
