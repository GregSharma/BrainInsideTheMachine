"""Scorer implementations: ways to compare two (T,d) feature tensors.

All scorers assume tensors are already fp32 on the same device.
No per-call dtype conversion. No CPU<->GPU shuffles.
"""
import math
import torch
import torch.nn.functional as F
from .base import Scorer


class CosineLast(Scorer):
    """Cosine of last-token features."""
    @property
    def name(self) -> str:
        return "cos_last"

    def __call__(self, chunk_feat: torch.Tensor, cue_feat: torch.Tensor, **kwargs) -> float:
        cv = chunk_feat[-1]
        qv = cue_feat[-1]
        return F.cosine_similarity(cv.unsqueeze(0), qv.unsqueeze(0), dim=-1).item()


class BilinearLast(Scorer):
    """Raw inner product of last-token features, scaled by 1/sqrt(d)."""
    @property
    def name(self) -> str:
        return "dot_last"

    def __call__(self, chunk_feat: torch.Tensor, cue_feat: torch.Tensor, **kwargs) -> float:
        d_h = kwargs.get("d_h", chunk_feat.shape[-1])
        return (chunk_feat[-1] @ cue_feat[-1]).item() / math.sqrt(d_h)


def _cross_attn_matrix(chunk_feat, cue_feat, d_h):
    """A[i,j] = <chunk_i, cue_j> / sqrt(d_h). Single matmul."""
    return chunk_feat @ cue_feat.T / math.sqrt(d_h)


class CrossMean(Scorer):
    """Mean of A[i,j] over all position pairs."""
    @property
    def name(self) -> str:
        return "cross_mean"

    def __call__(self, chunk_feat: torch.Tensor, cue_feat: torch.Tensor, **kwargs) -> float:
        d_h = kwargs.get("d_h", chunk_feat.shape[-1])
        return _cross_attn_matrix(chunk_feat, cue_feat, d_h).mean().item()


class CrossMax(Scorer):
    """Max over A[i,j] — best-aligned pair of positions."""
    @property
    def name(self) -> str:
        return "cross_max"

    def __call__(self, chunk_feat: torch.Tensor, cue_feat: torch.Tensor, **kwargs) -> float:
        d_h = kwargs.get("d_h", chunk_feat.shape[-1])
        return _cross_attn_matrix(chunk_feat, cue_feat, d_h).max().item()


class CrossSoftMax(Scorer):
    """(1/n) sum_i logsumexp_j A[i,j] — log-partition of row-wise attention."""
    @property
    def name(self) -> str:
        return "cross_softmax"

    def __call__(self, chunk_feat: torch.Tensor, cue_feat: torch.Tensor, **kwargs) -> float:
        d_h = kwargs.get("d_h", chunk_feat.shape[-1])
        A = _cross_attn_matrix(chunk_feat, cue_feat, d_h)
        return torch.logsumexp(A, dim=-1).mean().item()


class CrossCosineMean(Scorer):
    """Mean of cosine similarities across the full position grid."""
    @property
    def name(self) -> str:
        return "cross_cos_mean"

    def __call__(self, chunk_feat: torch.Tensor, cue_feat: torch.Tensor, **kwargs) -> float:
        cv = F.normalize(chunk_feat, dim=-1)
        qv = F.normalize(cue_feat, dim=-1)
        return (cv @ qv.T).mean().item()


class CrossCosineMax(Scorer):
    """Max cosine over all position pairs."""
    @property
    def name(self) -> str:
        return "cross_cos_max"

    def __call__(self, chunk_feat: torch.Tensor, cue_feat: torch.Tensor, **kwargs) -> float:
        cv = F.normalize(chunk_feat, dim=-1)
        qv = F.normalize(cue_feat, dim=-1)
        return (cv @ qv.T).max().item()


class CrossTopK(Scorer):
    """Mean of top-k cosine similarities across all (i,j) — robust max."""
    def __init__(self, k: int = 5):
        self.k = k

    @property
    def name(self) -> str:
        return f"cross_top{self.k}"

    def __call__(self, chunk_feat: torch.Tensor, cue_feat: torch.Tensor, **kwargs) -> float:
        cv = F.normalize(chunk_feat, dim=-1)
        qv = F.normalize(cue_feat, dim=-1)
        flat = (cv @ qv.T).flatten()
        k = min(self.k, flat.shape[0])
        return flat.topk(k).values.mean().item()
