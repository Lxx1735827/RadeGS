import torch
from typing import Optional


def depth_order_weight_schedule(iteration: int, schedule: str = "default"):
    if schedule == "default":
        lambda_depth_order = 0.
        if iteration > 3_000:
            lambda_depth_order = 1.
        if iteration > 7_000:
            lambda_depth_order = 0.1
        if iteration > 15_000:
            lambda_depth_order = 0.01
        if iteration > 20_000:
            lambda_depth_order = 0.001
        if iteration > 25_000:
            lambda_depth_order = 0.0001

    elif schedule == "strong":
        lambda_depth_order = 1.

    elif schedule == "weak":
        lambda_depth_order = 0.
        if iteration > 3_000:
            lambda_depth_order = 0.1

    elif schedule == "none":
        lambda_depth_order = 0.

    else:
        raise ValueError(f"Invalid schedule: {schedule}")

    return lambda_depth_order


import torch
from typing import Optional


def compute_depth_order_loss(
        depth: torch.Tensor,
        prior_depth: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        scene_extent: float = 1.0,
        max_pixel_shift_ratio: float = 0.05,
        normalize_loss: bool = True,
        log_space: bool = False,
        log_scale: float = 20.0,
        reduction: str = "mean",
        debug: bool = False,
):
    """
    Pairwise depth order consistency loss with strong NaN / Inf protection.
    """

    depth = depth.squeeze()
    prior_depth = prior_depth.squeeze()

    if mask is not None:
        mask = mask.squeeze().float()

    H, W = depth.shape
    device = depth.device

    # ============================================================
    # 1️⃣ HARD NaN / Inf CLEANING (最重要的一步)
    # ============================================================
    depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    prior_depth = torch.nan_to_num(prior_depth, nan=0.0, posinf=0.0, neginf=0.0)

    # 同时把非有限值从 mask 中剔除
    finite_mask = torch.isfinite(depth) & torch.isfinite(prior_depth)
    if mask is not None:
        mask = mask * finite_mask.float()
    else:
        mask = finite_mask.float()

    scene_extent = float(scene_extent)
    if scene_extent < 1e-6:
        scene_extent = 1e-6

    pixel_coords = torch.stack(
        torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        ),
        dim=-1
    ).view(-1, 2)

    max_pixel_shift = max(round(max_pixel_shift_ratio * max(H, W)), 1)

    pixel_shifts = torch.randint(
        -max_pixel_shift,
        max_pixel_shift + 1,
        pixel_coords.shape,
        device=device
    )

    shifted_pixel_coords = (pixel_coords + pixel_shifts).clamp(
        min=torch.tensor([0, 0], device=device),
        max=torch.tensor([H - 1, W - 1], device=device)
    )

    shifted_depth = depth[
        shifted_pixel_coords[:, 0],
        shifted_pixel_coords[:, 1]
    ].view(H, W)

    shifted_prior_depth = prior_depth[
        shifted_pixel_coords[:, 0],
        shifted_pixel_coords[:, 1]
    ].view(H, W)

    shifted_mask = mask[
        shifted_pixel_coords[:, 0],
        shifted_pixel_coords[:, 1]
    ].view(H, W)

    # pair-wise valid mask（两边都有效）
    valid_mask = (mask * shifted_mask).detach()

    diff = (depth - shifted_depth) / scene_extent
    prior_diff = (prior_depth - shifted_prior_depth) / scene_extent

    if normalize_loss:
        # sign(prior_diff)，但防止除 0
        prior_diff = prior_diff / prior_diff.detach().abs().clamp(min=1e-6)

    # 只惩罚顺序相反的情况
    depth_order_loss = - (diff * prior_diff).clamp(max=0.0)

    depth_order_loss = depth_order_loss * valid_mask

    # 再次防 NaN（mask 不能阻止 NaN 传播）
    depth_order_loss = torch.nan_to_num(
        depth_order_loss, nan=0.0, posinf=0.0, neginf=0.0
    )

    if log_space:
        depth_order_loss = depth_order_loss.clamp(min=0.0)
        depth_order_loss = torch.log1p(log_scale * depth_order_loss)

    if reduction == "mean":
        denom = valid_mask.sum().clamp(min=1.0)
        depth_order_loss = depth_order_loss.sum() / denom
    elif reduction == "sum":
        depth_order_loss = depth_order_loss.sum()
    elif reduction == "none":
        pass
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

    if not torch.isfinite(depth_order_loss):
        depth_order_loss = depth_order_loss.new_tensor(0.0)

    if debug:
        return {
            "loss": depth_order_loss,
            "valid_mask_sum": valid_mask.sum(),
            "diff": diff,
            "prior_diff": prior_diff,
        }

    return depth_order_loss



def sample_pixel_pairs(mask, num_pairs):
    """
    从掩码中随机采样像素对
    mask: H×W bool tensor
    return: (N, 2) index pairs in flattened index
    """
    # 展平，找到所有为true的像素的索引
    idx = torch.nonzero(mask.flatten(), as_tuple=False).squeeze(1)
    if idx.numel() < 2:
        return None
    # 随机采样
    perm = torch.randint(0, idx.numel(), (num_pairs * 2,), device=mask.device)
    u = idx[perm[:num_pairs]]
    v = idx[perm[num_pairs:]]
    return u, v

def depth_order_loss_(
    pred_depth,     # rendered_expected_depth (H×W)
    gt_depth,       # MoGe depth (H×W)
    mask,           # valid mask (H×W)
    num_pairs=8192,
    tau=0.02
):
    """
    tau: 相对深度阈值（后面我会解释怎么定）
    """
    if pred_depth.dim() == 3:
        pred_depth = pred_depth.squeeze(0)
    if gt_depth.dim() == 3:
        gt_depth = gt_depth.squeeze(0)
    if mask.dim() == 3:
        mask = mask.squeeze(0)
    device = pred_depth.device
    H, W = pred_depth.shape
    pred = pred_depth.flatten()
    gt = gt_depth.flatten()

    pairs = sample_pixel_pairs(mask, num_pairs)
    if pairs is None:
        return torch.tensor(0.0, device=device)

    u, v = pairs

    # MoGe depth difference
    d_gt = gt[u] - gt[v]

    # ordinal label
    label = torch.zeros_like(d_gt)
    label[d_gt > tau] = 1.0
    label[d_gt < -tau] = -1.0

    valid = label != 0
    if valid.sum() < 16:
        return torch.tensor(0.0, device=device)

    d_pred = pred[u] - pred[v]

    # logistic ranking loss
    loss = torch.log1p(torch.exp(-label[valid] * d_pred[valid]))

    return loss.mean()