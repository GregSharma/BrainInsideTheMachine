"""Retrieval framework for the BITM project.

Abstract objects:
  Feature   — maps (text, model, layer) -> tensor of shape (n_tokens, d)
  Scorer    — maps (chunk_feature_tensor, cue_feature_tensor) -> scalar
  Dataset   — produces (positives, negatives) lists of strings

Build features and scorers as composable units. Experiments are just
(feature x scorer x dataset x layer) loops.
"""
from .base import Feature, Scorer, ScoredResult
from .pipeline import encode_bare, run_grid, auc
