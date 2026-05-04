# Research Landscape Assessment
## Generated March 5, 2026 via Perplexity deep research

---

## 1. ARE WE SCOOPED?

**No.** Nobody has our result. Here's how close the field gets:

| Paper | What they show | What they DON'T show |
|-------|---------------|---------------------|
| LSAR (arXiv:2401.05792, 2024) | Low-rank subspace encodes language identity; nullspace improves cross-lingual tasks | No reasoning subspace, no R^2 bridge, syntax not topic |
| Activation Patching (arXiv:2411.08745, ACL 2025) | Cross-lingual concept patching works; concepts factorize from language | No explicit subspace geometry, no dimensionality, no R^2 |
| Layer Swapping for Math (arXiv:2410.01335, ICLR 2025) | Math reasoning in middle layers, language in early/late | Behavioral not geometric; no shared subspace identification |
| LENS / Language-Reasoning Disentanglement (OpenReview 2025) | Orthogonal decomposition into lang-agnostic and lang-specific | No topic clustering, no cross-model bridge |
| NLLB-200 Conceptual Store (arXiv:2603.02258, 2026) | Mean-centering reveals cross-lingual concept clustering | Translation concepts only, not reasoning; 1.19x improvement not R^2=0.976 |
| Shared Geometry of Difficulty (arXiv:2601.12731, 2025) | Language-agnostic "difficulty" direction in early layers | Single direction, not 20-dim subspace; difficulty not content |
| Language Steering (arXiv:2602.02326, 2026) | Task-agnostic steering vectors cluster by language family | Language-conditioned directions, not language-invariant reasoning |
| Platonic Representation Hypothesis (Huh et al., ICML 2024) | Models converge to shared "platonic" representation | Theoretical position paper; no empirical subspace extraction |

**Our unique contributions (none replicated elsewhere):**
1. Explicit 20-dim language-agnostic reasoning subspace with constructive identification
2. R^2 = 0.976 cross-lingual bridge (nobody else quantifies like this)
3. Topic-rather-than-language clustering in intermediate layers (unproven elsewhere)
4. Causal proof via activation patching double dissociation (Z-patch invisible, Z-perp destructive)
5. Cross-model gradient: 4 labs, clean R^2 correlation with bilingual integration depth
6. 7-language generalization with 4.24x NN over random baseline

**Framing:** The field has pieces. LSAR found language is low-rank. Patching proved concepts are shared. Layer-swapping showed math reasoning is localized. Platonic Rep Hypothesis predicts convergence. **We are the first to put it all together with precise geometry.**

---

## 2. TARGET AUDIENCE

### Tier 1: Highest-value contacts (DM after arxiv)

**Chris Olah** (@ch402, uncertain if active)
- Anthropic co-founder, Interpretability Research Lead
- Built the field: Toy Models of Superposition, Towards Monosemanticity
- History of championing visually striking, conceptually clean outsider work
- Fig 1 (hero t-SNE) is exactly his aesthetic
- **Risk:** Very busy, may not check DMs

**Neel Nanda** (@NeelNanda5)
- Mechanistic Interpretability Lead at Google DeepMind
- Most visible community hub; explicitly encourages independent projects
- Active on X, retweets independent work
- **Best first contact for visibility**

**Minyoung Huh, Brian Cheung, Tongzhou Wang, Phillip Isola** (all MIT)
- Authors of Platonic Representation Hypothesis (ICML 2024)
- Our work is direct empirical evidence for their thesis
- They predicted convergent representations; we found them
- AND we found where they break (across non-bilingual models)
- **They would want to cite us**

### Tier 2: Strong relevance

**Emmanuel Ameisen** (Anthropic)
- Discussed "multilingual circuits" in 2025 Latent Space interview
- Anthropic's public voice on cross-lingual representations
- High-value because he's already thinking about this exact problem

**Christina Lu, Jack Gallagher** (Anthropic)
- "The Assistant Axis" paper -- low-dim persona directions across models
- Our Z is to reasoning what their assistant axis is to persona
- Methodological kinship

**David Bai** (@dav1d_bai, Cornell)
- "Harnessing the Universal Geometry of Embeddings"
- Active on X, threads his own research, engages with non-affiliated accounts
- Strong Platonic Representation Hypothesis advocate

### Tier 3: Community amplifiers

**@mlwires** - curation account, amplifies geometry/embedding papers
**Beren Millidge** - "Deep learning models are secretly (almost) linear" blog
**EleutherAI Discord** - active independent research community

---

## 3. RELATED WORK FOR PAPER

### Must-cite:
- Huh et al. 2024 - Platonic Representation Hypothesis (theoretical framework)
- arXiv:2411.08745 - Activation patching cross-lingual concepts (our causal method)
- arXiv:2410.01335 - Layer swapping for math reasoning (localization claim)
- arXiv:2401.05792 - LSAR low-rank language subspaces (complementary finding)
- Anthropic "Tracing the thoughts of a large language model" (universal language of thought)
- Toy Models of Superposition (linear representation hypothesis lineage)

### Should-cite:
- arXiv:2603.02258 - NLLB conceptual store (closest geometric work)
- arXiv:2601.12731 - Shared geometry of difficulty (language-agnostic meta-variable)
- arXiv:2602.02326 - Language steering vectors (language-conditioned geometry)
- OpenReview 2025 - LENS / Language-Reasoning Disentanglement

### Contrarian / preemptive:
- Alignment-Transfer Dissociation work (geometric alignment != functional transfer)
- mOthello synthetic task (high alignment scores, failed transfer)
- "Onion" representations violating strong LRH

---

## 4. APPLICATION LANDSCAPE

### Z-aware quantization
- **FASC (2026)** is closest: Fisher-aligned subspace compression
- **DiaQ (ICLR 2026)** preserves activation direction under quantization
- **FreeAct (2026)** reshapes activations for quantization-friendly geometry
- **MoHD (2025)** routes tokens to sub-dimensions dynamically
- **GAP:** Nobody does explicit functional-subspace mixed precision (FP16 for Z, INT4 for Z-perp)
- **Our advantage:** We HAVE the subspace. They're trying to find it via Fisher/Hessian. We found it via cross-lingual invariance.

### Speculative decoding / model splicing
- **Lyanna (2025)** builds hidden-state chains, reuses rejected states, 3.3x speedup
- **EAGLE/SpecInfer** use tree verification with hidden states
- **BRIDGE (2025)** projects across modalities (vision->text)
- **GAP:** Nobody projects small-LLM hidden states into large-LLM space for inference
- **Our advantage:** R^2=0.94 between 1.5B and 3B means this projection EXISTS and is linear

### Z-preserving training
- No direct literature on auxiliary losses preserving cross-lingual subspace
- Closest: knowledge distillation, representation regularization
- **Wide open**

---

## 5. URGENCY ASSESSMENT

**Low scooping risk for the core result.** The specific combination (low-dim Z + R^2 bridge + causal patching + cross-model gradient + 7-language generalization) is ours alone. Nobody else has all the pieces.

**Medium risk on individual components:**
- NLLB conceptual store (arXiv:2603.02258, Feb 2026) is moving toward geometric proof of cross-lingual clustering
- LENS/Disentanglement work (OpenReview 2025) is independently arriving at orthogonal decomposition
- Someone at Anthropic (Ameisen) is thinking about multilingual circuits

**Recommendation:** Paper first, applications second. The paper protects priority. The applications (quantization, splicing, training objectives) are follow-up papers that cite paper 1.

**Timeline:** Get on arxiv within 2-3 weeks. The field is converging.
