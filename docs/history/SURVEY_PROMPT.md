# BITM Survey Prompt
# Run this in a fresh claude session in the BrainInsideTheMachine dir.
# Paste everything below the line.
---

Load state for BrainInsideTheMachine. Then search memory for ALL semantic memories in this project — use search_memory with broad queries: "phase results findings", "entanglement language", "MLP direction flip", "cross-lingual", "Z subspace", "patching causal", "Ghost of Neumann", "von Neumann paper", "MOAMS probe", "centroid", "residual stream", "token budget", "generation strategy", "early exit", "attention heads", "anti-heads", "3B artifact", "8B comparison", "convergence". Cast a wide net — I'd rather you over-retrieve than miss threads.

Also load the last 5 state snapshots for this project via list_state_snapshots then load_state_snapshots.

I've been away for 3 weeks. 47 sessions of research across 3 months. The markdown docs in this directory are STALE — trust the orchestrator memories and state snapshots over any .md file. The memories are the ground truth.

Give me:

1. CONFIRMED FINDINGS — things we tested and know are true. What the evidence was. Not hypotheses — results. Include the experiment name/number if you can find it.

2. THINGS THAT BROKE — results that contradicted what we predicted. What we expected vs what actually happened. These are often more interesting than the confirmations.

3. DANGLING THREADS — experiments proposed but never run. Questions raised but never answered. Ideas that got lost in context clears or session ends. For each: is it still worth pulling on, or did later work make it irrelevant?

4. THE WEIRD SHIT — anything anomalous, unexplained, or surprising that doesn't fit neatly into the above. The Ghost of Neumann goes here (read GHOST_OF_NEUMANN.md for the full forensics — that one IS up to date). Anything else like it.

5. WHAT CONNECTS — if you see patterns across findings that we never explicitly connected, say so. You have the full memory graph. Use it.

Don't write a paper outline. Don't frame for publication. This is a curiosity inventory. What do we know, what don't we know, and what's still worth knowing.

Write the output to SURVEY_RESULTS.md in this directory.
