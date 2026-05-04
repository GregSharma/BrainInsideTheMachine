"""Shared utilities for transformer weight analysis."""

import numpy as np
import torch


def effective_rank(W):
    """Entropy-based effective rank: r_eff = exp(H(sigma_hat))."""
    S = torch.linalg.svdvals(W.float().cpu())
    sigma_hat = S / S.sum()
    sigma_hat = sigma_hat[sigma_hat > 0]
    H = -(sigma_hat * sigma_hat.log()).sum().item()
    return np.exp(H)


def get_attn_subspace(model, layer_idx, h, GQA, d, head_idx, k=20):
    """Get top-k right singular vectors of attention kernel W_Q_h^T @ W_K_h.

    Returns (k, d) tensor of orthonormal row vectors spanning the dominant
    subspace that this head queries in.
    """
    layer = model.model.layers[layer_idx]
    d_head = d // h
    n_kv_groups = h // GQA
    kv_head_idx = head_idx // n_kv_groups

    W_Q_h = layer.self_attn.q_proj.weight.data[
        head_idx * d_head : (head_idx + 1) * d_head, :
    ].float().cpu()
    W_K_h = layer.self_attn.k_proj.weight.data[
        kv_head_idx * d_head : (kv_head_idx + 1) * d_head, :
    ].float().cpu()

    kernel = W_Q_h.T @ W_K_h  # (d, d)
    _, S, Vh = torch.linalg.svd(kernel)
    return Vh[:k, :]  # (k, d)


def subspace_similarity(V1, V2):
    """Grassmann similarity: mean squared cosine of principal angles.

    1.0 = identical subspaces, 0.0 = orthogonal.
    """
    M = V1 @ V2.T  # (k, k)
    svals = torch.linalg.svdvals(M)
    return (svals ** 2).mean().item()


def get_model_dims(model):
    """Extract key dimensions from model config."""
    cfg = model.config
    return {
        "L": cfg.num_hidden_layers,
        "d": cfg.hidden_size,
        "d_ff": cfg.intermediate_size,
        "h": cfg.num_attention_heads,
        "GQA": cfg.num_key_value_heads,
        "V": cfg.vocab_size,
        "d_head": cfg.hidden_size // cfg.num_attention_heads,
        "d_kv": (cfg.hidden_size // cfg.num_attention_heads) * cfg.num_key_value_heads,
    }
