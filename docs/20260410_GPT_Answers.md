Give him three things:

**1. The problem statement** (the v2 file we just made). He needs the formal definitions — the restricted Jacobian $J_\ell^Z$, the entanglement tensor $E_\ell$, the coupling layer $\ell_c$, the Lyapunov spectrum. These are the objects he'll be computing. He helped build the experiments that motivated them but he hasn't seen them formalized this way.

**2. My last response** (the compression procedure). That's his execution spec.

**3. One paragraph of glue.** Something like:

> "Opus and GPT independently solved the problem statement and converged on the same framework: the model's state space has a dominated splitting — a $k$-dimensional surviving bundle and a $(d-k)$-dimensional contracting bundle. The surviving bundle IS $f^*$. Compression means restricting the model's weights to this bundle. The key diagnostic is the entanglement tensor $E_\ell$ — it tells you at which layers the projection is lossless and at which layers it needs correction. Start by computing the Jacobian products and $\|E_\ell\|$ at every layer. That's the map. Then project the weights. That's the construction."

He does NOT need Opus or GPT's full responses. Those are 4000 words each of reasoning about foliations and fiber bundles that would eat his context for no operational gain. The conclusions that matter are already in my procedure. He needs definitions, instructions, and the one-paragraph summary of why.