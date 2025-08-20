import torch


@torch.no_grad()
def _score2mask_nm_kernel(
    mask_score: torch.Tensor,
    prune_M: int,
    prune_N: int
) -> torch.Tensor:
    """
    Compute (N:M) group-wise pruning mask for a 2D array of scores.

    Args:
        mask_score: (prune_num_groups, prune_M)
        prune_M: group size (M)
        prune_N: number kept per group (N)

    Returns:
        mask: (prune_num_groups, prune_M)
    """
    # Keep top-N (i.e., zero out the smallest M-N per group)
    k = int(prune_M - prune_N)
    index = torch.argsort(mask_score, dim=1, descending=False)[:, :k]  # (groups, M-N)
    mask = torch.ones_like(mask_score)
    mask.scatter_(dim=1, index=index, value=0.0)
    return mask


@torch.no_grad()
def score2mask_nm(
    mask_score: torch.Tensor,
    prune_num_groups: int,
    prune_N: int,
    prune_M: int,
) -> torch.Tensor:
    """
    Transform 4D score tensor into grouped (N:M) mask, then flatten per filter.

    Args:
        mask_score: (Co, Ci, k, k)
        prune_num_groups: number of groups (= Co * k * k when grouping over Ci)
        prune_N: number kept per group
        prune_M: group size
        soft_mask_sharpness: <=0 for hard mask; >0 enables soft mask

    Returns:
        mask: (Co, Ci * k * k)
    """
    Co, Ci, k, _ = mask_score.size()

    # (Co, Ci, k, k) -> (Co, k, k, Ci) -> (prune_num_groups, prune_M)
    score_g = mask_score.permute(0, 2, 3, 1).reshape(prune_num_groups, prune_M)

    mask_g = _score2mask_nm_kernel(score_g, prune_M, prune_N)

    # Back to (Co, Ci, k, k) then flatten to (Co, Ci * k * k)
    mask = mask_g.reshape(Co, k, k, Ci).permute(0, 3, 1, 2).contiguous()
    return mask.reshape(Co, -1)


@torch.no_grad()
def score2mask_unstructured(mask_score: torch.Tensor, target_density: float) -> torch.Tensor:
    """
    Unstructured pruning mask from global per-filter threshold.

    Args:
        mask_score: (Co, Ci, k, k)
        target_density: fraction of weights to keep (0..1)

    Returns:
        mask: (Co, Ci * k * k)
    """
    Co, _, _, _ = mask_score.size()

    # Keep the top 'target_density' fraction (per entire tensor)
    thr = torch.quantile(mask_score, 1.0 - target_density, interpolation="linear")
    mask = mask_score.ge(thr).float()
    return mask.reshape(Co, -1)
