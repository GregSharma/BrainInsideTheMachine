"""Abstract base classes for the retrieval framework."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class EncodingCache:
    """Per-text cache of everything we might extract from one forward pass.

    Attributes:
      hidden_states: tensor of shape (L+1, T, d) — all layers, all tokens
      n_layers: int — number of transformer blocks (L)
      n_tokens: int — sequence length (T)
      d: int — hidden dim
    """
    hidden_states: torch.Tensor
    n_layers: int
    n_tokens: int
    d: int


class Feature(ABC):
    """Maps an EncodingCache + layer index -> (T, d') tensor of token features.

    Subclasses pick what object to extract: raw hidden states, deltas,
    innovations, Q/K/V projections, sequence-innovation residuals, etc.
    """

    @abstractmethod
    def extract(self, cache: EncodingCache, layer: int, model=None) -> torch.Tensor:
        """Returns (T, d') tensor."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class Scorer(ABC):
    """Maps (chunk_feature_tensor, cue_feature_tensor) -> scalar score.

    Both tensors are (T, d). Scorer decides how to aggregate across positions:
    last-token, mean, max, soft-max, bilinear, etc.
    """

    @abstractmethod
    def __call__(self, chunk_feat: torch.Tensor, cue_feat: torch.Tensor, **kwargs) -> float:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


@dataclass
class ScoredResult:
    """One scored chunk-cue pair."""
    chunk_idx: int
    cue_idx: int
    score: float
    label: int  # 1 = relevant, 0 = irrelevant
