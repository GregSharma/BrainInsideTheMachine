"""Feature implementations: things you can extract from an encoded text."""
import torch
from .base import Feature, EncodingCache


def _per_sample_gram_schmidt(series: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Project each element of a series onto the span of the previous k elements.

    series: (N, d) — sequence of vectors
    Returns (innovation, predictable) each (N, d).

    Vectorized: O(1) Python loop iterations (only the k boundary tokens).
    All N-k full-history projections are done in one batched QR call.
    """
    N, d = series.shape
    eps = series.clone()  # eps[0] = series[0] by default; rest overwritten below
    pred = torch.zeros_like(series)

    if N <= 1:
        return eps, pred

    # Boundary: tokens i in [1, min(k, N)) have variable-length history of size i.
    # k is small (typically 2-8), so this loop is O(k) iterations max — not O(N).
    for i in range(1, min(k, N)):
        hist = series[:i]                        # (i, d)
        Q, _ = torch.linalg.qr(hist.T)           # (d, i)
        coef = Q.T @ series[i]                   # (i,)
        proj = Q @ coef
        pred[i] = proj
        eps[i] = series[i] - proj

    # Bulk: tokens i in [k, N) all have full history of size k. One batched QR.
    if N > k:
        # windows[t] = series[t : t+k], for t in [0, N-k)  -> shape (N-k, k, d)
        # series.unfold(0, k, 1) -> (N-k+1, d, k); transpose last two -> (N-k+1, k, d)
        # take first N-k slices to get histories for targets at indices k..N-1
        windows = series.unfold(0, k, 1).transpose(-1, -2)[:N - k]   # (N-k, k, d)
        targets = series[k:N]                                         # (N-k, d)
        Qb, _ = torch.linalg.qr(windows.transpose(-1, -2))            # (N-k, d, k)
        coef = torch.einsum('bdk,bd->bk', Qb, targets)                # (N-k, k)
        proj = torch.einsum('bdk,bk->bd', Qb, coef)                   # (N-k, d)
        pred[k:N] = proj
        eps[k:N] = targets - proj

    return eps, pred


class HiddenState(Feature):
    """Raw residual stream h_l at every token."""
    @property
    def name(self) -> str:
        return "h"

    def extract(self, cache: EncodingCache, layer: int, model=None) -> torch.Tensor:
        # hidden_states[layer] is shape (T, d). layer in {0..L}
        return cache.hidden_states[layer]


class Delta(Feature):
    """Per-layer increment: delta_l = h_{l+1} - h_l, at every token."""
    @property
    def name(self) -> str:
        return "delta"

    def extract(self, cache: EncodingCache, layer: int, model=None) -> torch.Tensor:
        if layer >= cache.n_layers:
            raise IndexError(f"delta requires layer < {cache.n_layers}, got {layer}")
        return cache.hidden_states[layer + 1] - cache.hidden_states[layer]


class DepthInnovation(Feature):
    """Innovation of delta_l across layers (per-token, Gram-Schmidt on depth history).

    For each token, compute the residual of delta_l after projecting onto
    the span of delta_{l-1}, ..., delta_{l-k}.
    """
    def __init__(self, k: int = 2):
        self.k = k

    @property
    def name(self) -> str:
        return f"eps_depth_k{self.k}"

    def extract(self, cache: EncodingCache, layer: int, model=None) -> torch.Tensor:
        # Stack deltas across layers for each token, do GS along depth
        if layer >= cache.n_layers:
            raise IndexError
        # Build depth-deltas at this token: shape (n_layers, d) per token
        deltas = cache.hidden_states[1:] - cache.hidden_states[:-1]  # (L, T, d)
        # Innovation of delta_l given depth history at each token
        result = torch.zeros(cache.n_tokens, cache.d, dtype=deltas.dtype)
        for t in range(cache.n_tokens):
            series = deltas[:, t, :]  # (L, d)
            eps, _ = _per_sample_gram_schmidt(series, k=self.k)
            result[t] = eps[layer]
        return result


class SequenceInnovation(Feature):
    """Innovation of h_l across tokens (Gram-Schmidt on token history).

    For each token t, project h_l^t onto span of h_l^{t-k}, ..., h_l^{t-1}.
    The residual is the per-token innovation at layer l.
    """
    def __init__(self, k: int = 4):
        self.k = k

    @property
    def name(self) -> str:
        return f"eps_seq_k{self.k}"

    def extract(self, cache: EncodingCache, layer: int, model=None) -> torch.Tensor:
        series = cache.hidden_states[layer]  # (T, d)
        eps, _ = _per_sample_gram_schmidt(series, k=self.k)
        return eps


class QProj(Feature):
    """Apply W_Q at layer l to a base feature."""
    def __init__(self, base: Feature):
        self.base = base

    @property
    def name(self) -> str:
        return f"Q({self.base.name})"

    def extract(self, cache: EncodingCache, layer: int, model=None) -> torch.Tensor:
        x = self.base.extract(cache, layer, model)
        with torch.no_grad():
            q_w = model.model.layers[layer].self_attn.q_proj.weight.data    # (d_q, d)
            return (x.to(model.device, dtype=q_w.dtype) @ q_w.T).float()


class KProj(Feature):
    """Apply W_K at layer l (with GQA head expansion to match Q dim)."""
    def __init__(self, base: Feature):
        self.base = base

    @property
    def name(self) -> str:
        return f"K({self.base.name})"

    def extract(self, cache: EncodingCache, layer: int, model=None) -> torch.Tensor:
        x = self.base.extract(cache, layer, model)
        n_heads = model.config.num_attention_heads
        n_kv = model.config.num_key_value_heads
        rep = n_heads // n_kv
        with torch.no_grad():
            k_w = model.model.layers[layer].self_attn.k_proj.weight.data    # (d_kv, d)
            k = (x.to(model.device, dtype=k_w.dtype) @ k_w.T).float()       # (T, d_kv)
            head_dim = k.shape[-1] // n_kv
            k = k.view(-1, n_kv, head_dim).repeat(1, rep, 1).reshape(-1, n_heads * head_dim)
        return k


class VProj(Feature):
    """Apply W_V at layer l (with GQA head expansion)."""
    def __init__(self, base: Feature):
        self.base = base

    @property
    def name(self) -> str:
        return f"V({self.base.name})"

    def extract(self, cache: EncodingCache, layer: int, model=None) -> torch.Tensor:
        x = self.base.extract(cache, layer, model)
        n_heads = model.config.num_attention_heads
        n_kv = model.config.num_key_value_heads
        rep = n_heads // n_kv
        with torch.no_grad():
            v_w = model.model.layers[layer].self_attn.v_proj.weight.data
            v = (x.to(model.device, dtype=v_w.dtype) @ v_w.T).float()
            head_dim = v.shape[-1] // n_kv
            v = v.view(-1, n_kv, head_dim).repeat(1, rep, 1).reshape(-1, n_heads * head_dim)
        return v
