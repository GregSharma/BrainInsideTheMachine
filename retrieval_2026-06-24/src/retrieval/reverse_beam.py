"""Reverse beam search: find the most likely prefix tokens given a chunk.

For a chunk x_1, ..., x_n, find argmax_{v_0} p(x_1, ..., x_n | v_0).
Brute force over vocab is feasible because we batch.

Beam-K extension: keep top-K v_0's, extend each backward by one token.
Complexity: O(K * |V| * forward_pass) per extension step.

Computational note:
  Single forward pass on Qwen2.5-3B at length n+1 with batch B ≈ 256 fits in 12GB.
  |V| / B = 152k / 256 = 594 batches. Each batch ~50ms on RTX 4070.
  Total: ~30s per chunk for length-1 reverse search.
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .base import EncodingCache


def chunk_log_likelihood_batched(
    prefix_ids: torch.Tensor,   # (B, P) — batch of prefix sequences
    chunk_ids: torch.Tensor,    # (T,) — single chunk, will be tiled
    model,
    batch_size: int = 64,
    show_progress: bool = False,
    progress_desc: str = "ll",
) -> torch.Tensor:
    """Returns (B,) tensor of log p(chunk | prefix_b) for each prefix.

    Memory-bounded by batch_size. Optional inner tqdm so callers don't
    suffer silent multi-minute waits on big vocab sweeps.
    """
    B, P = prefix_ids.shape
    T = chunk_ids.shape[0]
    device = next(model.parameters()).device
    chunk_ids = chunk_ids.to(device)
    out = torch.zeros(B, device=device)

    iterator = range(0, B, batch_size)
    if show_progress:
        from tqdm import tqdm as _tqdm
        iterator = _tqdm(list(iterator), desc=progress_desc, leave=False, mininterval=0.3)

    for start in iterator:
        end = min(start + batch_size, B)
        bsz = end - start
        prefix_batch = prefix_ids[start:end].to(device)
        chunk_batch = chunk_ids.unsqueeze(0).expand(bsz, -1)
        full = torch.cat([prefix_batch, chunk_batch], dim=1)

        with torch.no_grad():
            logits = model(full).logits

        pred_logits = logits[:, P - 1 : P + T - 1, :]
        log_probs = F.log_softmax(pred_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, chunk_batch.unsqueeze(-1)).squeeze(-1)
        out[start:end] = token_log_probs.sum(dim=-1)

    return out


def reverse_search_length1(
    chunk_text: str,
    tok,
    model,
    top_k: int = 50,
    batch_size: int = 64,
    candidate_vocab: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find top-K most likely single-token prefixes v_0 for a chunk.

    Returns:
      top_ids: (top_k,) token ids
      top_logp: (top_k,) log p(chunk | v_0) for each

    candidate_vocab: optional (V',) tensor of token ids to score.
                     If None, uses the full vocab (slow on small GPUs).
    """
    chunk_ids = tok(chunk_text, return_tensors="pt").input_ids[0]

    if candidate_vocab is None:
        V = model.config.vocab_size
        candidate_vocab = torch.arange(V)

    Vp = candidate_vocab.shape[0]
    prefix_ids = candidate_vocab.unsqueeze(-1)  # (Vp, 1)

    log_p = chunk_log_likelihood_batched(
        prefix_ids, chunk_ids, model, batch_size=batch_size,
        show_progress=True, progress_desc="vocab",
    )
    top = log_p.topk(top_k)
    return candidate_vocab[top.indices.cpu()], top.values.cpu()


def reverse_beam_extend(
    chunk_text: str,
    tok,
    model,
    current_beams: torch.Tensor,        # (K, P) — current top-K prefixes
    current_logp: torch.Tensor,         # (K,)
    beam_k: int = 50,
    candidate_vocab: torch.Tensor = None,
    batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extend each of K beams by one token leftward.

    For each beam b and each candidate v, compute log p(chunk | [v ; beam_b]).
    Keep top-K (v, b) pairs.
    """
    chunk_ids = tok(chunk_text, return_tensors="pt").input_ids[0]
    K, P = current_beams.shape
    if candidate_vocab is None:
        candidate_vocab = torch.arange(model.config.vocab_size)
    Vp = candidate_vocab.shape[0]

    # Build all (K * Vp) candidate prefixes: prepend each v to each beam
    # Shape: (K * Vp, P + 1)
    beams_exp = current_beams.unsqueeze(1).expand(-1, Vp, -1).reshape(K * Vp, P)
    vocab_exp = candidate_vocab.unsqueeze(0).expand(K, -1).reshape(K * Vp, 1)
    candidates = torch.cat([vocab_exp, beams_exp], dim=1)  # (K*Vp, P+1)

    log_p = chunk_log_likelihood_batched(candidates, chunk_ids, model, batch_size=batch_size)
    top = log_p.topk(beam_k)
    return candidates[top.indices.cpu()], top.values.cpu()


def reverse_prefix_feature(
    chunk_text: str,
    tok,
    model,
    layer: int,
    top_k: int = 50,
    extend_steps: int = 0,
    candidate_vocab: torch.Tensor = None,
    batch_size: int = 64,
) -> torch.Tensor:
    """Compute a feature vector that represents 'what context this chunk wants before it'.

    Method:
      1. Find top_k single-token prefixes by log-likelihood of chunk.
      2. (Optional) Extend to longer prefixes via beam_extend.
      3. Weight each prefix's last-token hidden state at `layer` by softmax(log p).
      4. Return weighted mean.

    Returns: (d,) tensor.
    """
    ids, logps = reverse_search_length1(
        chunk_text, tok, model, top_k=top_k,
        candidate_vocab=candidate_vocab, batch_size=batch_size,
    )
    beams = ids.unsqueeze(-1)  # (K, 1)
    for _ in range(extend_steps):
        beams, logps = reverse_beam_extend(
            chunk_text, tok, model, beams, logps,
            beam_k=top_k, candidate_vocab=candidate_vocab, batch_size=batch_size,
        )

    # Now encode each prefix and read its last-token hidden state at `layer`
    device = next(model.parameters()).device
    weights = F.softmax(logps, dim=0).to(device)
    out = None
    with torch.no_grad():
        for i in range(beams.shape[0]):
            prefix_ids = beams[i].unsqueeze(0).to(device)
            h = model(prefix_ids, output_hidden_states=True).hidden_states[layer][0, -1, :].float()
            if out is None:
                out = torch.zeros_like(h)
            out += weights[i] * h
    return out.cpu()
