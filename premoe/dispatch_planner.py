"""Dispatch planning: predict which tokens need cross-GPU communication."""

import torch


def compute_dispatch_plan(
    topk_ids: torch.Tensor,
    num_experts: int,
    ep_size: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute which tokens need to be sent to peer GPU(s).

    Args:
        topk_ids: [N, topk] expert indices per token.
        num_experts: Total number of experts.
        ep_size: Expert parallelism degree.
        rank: Current GPU rank.

    Returns:
        peer_token_indices: 1-D tensor of token indices that need peer dispatch.
        needs_peer: [N] bool tensor indicating which tokens need peer.
    """
    experts_per_gpu = num_experts // ep_size
    gpu_dest = topk_ids // experts_per_gpu  # [N, topk]
    needs_peer = (gpu_dest != rank).any(dim=-1)  # [N]
    peer_token_indices = torch.where(needs_peer)[0]
    return peer_token_indices, needs_peer


def pack_tokens(
    hidden: torch.Tensor,
    token_indices: torch.Tensor,
) -> torch.Tensor:
    """Pack selected tokens into a contiguous buffer for sending.

    Args:
        hidden: [N, D] hidden states.
        token_indices: 1-D tensor of indices to pack.

    Returns:
        Contiguous tensor of shape [len(token_indices), D].
    """
    if len(token_indices) == 0:
        return torch.empty(
            0, hidden.shape[-1], dtype=hidden.dtype, device=hidden.device
        )
    return hidden[token_indices].contiguous()


def verify_dispatch(
    pred_ids: torch.Tensor,
    true_ids: torch.Tensor,
    num_experts: int,
    ep_size: int,
    rank: int,
) -> torch.Tensor:
    """Compare predicted vs true routing at GPU-dispatch level.

    Returns indices of tokens with mismatched GPU destinations.
    """
    experts_per_gpu = num_experts // ep_size
    pred_gpu = pred_ids // experts_per_gpu
    true_gpu = true_ids // experts_per_gpu

    pred_needs_peer = (pred_gpu != rank).any(dim=-1)
    true_needs_peer = (true_gpu != rank).any(dim=-1)

    mismatch = pred_needs_peer != true_needs_peer
    return torch.where(mismatch)[0]
