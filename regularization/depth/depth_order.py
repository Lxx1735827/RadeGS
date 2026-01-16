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


def compute_depth_order_loss(
        depth: torch.Tensor,
        prior_depth: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        scene_extent: float = 1.,
        max_pixel_shift_ratio: float = 0.05,
        normalize_loss: bool = True,
        log_space: bool = False,
        log_scale: float = 20.,
        reduction: str = "mean",
        debug: bool = False,
):
    """Compute a loss encouraging pixels in 'depth' to have the same relative depth order as in 'prior_depth'.
    This loss does not require prior depth maps to be multi-view consistent nor to have accurate relative scale.

    Args:
        mask:
        depth (torch.Tensor): A tensor of shape (H, W), (H, W, 1) or (1, H, W) containing the depth values.
        prior_depth (torch.Tensor): A tensor of shape (H, W), (H, W, 1) or (1, H, W) containing the prior depth values.
        scene_extent (float): The extent of the scene used to normalize the loss and make the loss invariant to the scene scale.
        max_pixel_shift_ratio (float, optional): The maximum pixel shift ratio. Defaults to 0.05, i.e. 5% of the image size.
        normalize_loss (bool, optional): Whether to normalize the loss. Defaults to True.
        reduction (str, optional): The reduction to apply to the loss. Can be "mean", "sum" or "none". Defaults to "mean".

    Returns:
        torch.Tensor: A scalar tensor.
            If reduction is "none", returns a tensor with same shape as depth containing the pixel-wise depth order loss.
    """
    # ---------- Shape handling ----------
    depth = depth.squeeze()
    prior_depth = prior_depth.squeeze()
    if mask is not None:
        mask = mask.squeeze().float()

    height, width = depth.shape
    device = depth.device

    # ---------- Pixel coordinates ----------
    pixel_coords = torch.stack(
        torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij"
        ),
        dim=-1
    ).view(-1, 2)

    # ---------- Random pixel shifts ----------
    max_pixel_shift = max(round(max_pixel_shift_ratio * max(height, width)), 1)
    pixel_shifts = torch.randint(
        -max_pixel_shift,
        max_pixel_shift + 1,
        pixel_coords.shape,
        device=device
    )

    shifted_pixel_coords = (pixel_coords + pixel_shifts).clamp(
        min=torch.tensor([0, 0], device=device),
        max=torch.tensor([height - 1, width - 1], device=device)
    )

    # ---------- Gather shifted values ----------
    shifted_depth = depth[
        shifted_pixel_coords[:, 0],
        shifted_pixel_coords[:, 1]
    ].view(height, width)

    shifted_prior_depth = prior_depth[
        shifted_pixel_coords[:, 0],
        shifted_pixel_coords[:, 1]
    ].view(height, width)

    # ---------- Mask handling ----------
    if mask is not None:
        shifted_mask = mask[
            shifted_pixel_coords[:, 0],
            shifted_pixel_coords[:, 1]
        ].view(height, width)

        # valid only if both pixels are valid
        valid_mask = (mask * shifted_mask).detach()
    else:
        valid_mask = torch.ones_like(depth)

    # ---------- Depth order loss ----------
    diff = (depth - shifted_depth) / scene_extent
    prior_diff = (prior_depth - shifted_prior_depth) / scene_extent

    if normalize_loss:
        prior_diff = prior_diff / prior_diff.detach().abs().clamp(min=1e-8)

    depth_order_loss = - (diff * prior_diff).clamp(max=0)

    if log_space:
        depth_order_loss = torch.log1p(log_scale * depth_order_loss)

    # apply mask
    depth_order_loss = depth_order_loss * valid_mask

    # ---------- Reduction ----------
    if reduction == "mean":
        denom = valid_mask.sum().clamp(min=1.0)
        depth_order_loss = depth_order_loss.sum() / denom
    elif reduction == "sum":
        depth_order_loss = depth_order_loss.sum()
    elif reduction == "none":
        pass
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

    # ---------- Debug ----------
    if debug:
        return {
            "depth_order_loss": depth_order_loss,
            "diff": diff,
            "prior_diff": prior_diff,
            "shifted_depth": shifted_depth,
            "shifted_prior_depth": shifted_prior_depth,
            "valid_mask": valid_mask,
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