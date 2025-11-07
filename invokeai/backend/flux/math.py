# Initially pulled from https://github.com/black-forest-labs/flux

import torch
from einops import rearrange
from torch import Tensor


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, attn_mask: Tensor | None = None) -> Tensor:
    q, k = apply_rope(q, k, pe)

    x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    device = pos.device
    pos_dtype = pos.dtype

    # Precompute scale and omega with the proper dtype up front
    if device.type == "mps":
        arange_dtype = torch.float32
    else:
        arange_dtype = torch.float64

    scale = torch.arange(0, dim, 2, dtype=arange_dtype, device=device) / dim
    omega = 1.0 / (theta**scale)

    # Compute cos/sin and stack directly for maximum reuse and parallelism
    # Replace einsum with broadcasting for efficiency
    # out shape (..., n, d), omega shape (d,), broadcast to (..., n, d)
    out = pos.unsqueeze(-1) * omega  # (..., n) x (d,) -> (..., n, d)
    # To ensure same memory layout/order as einsum, expand on demand
    # Compute sin and cos once
    cos_out = torch.cos(out)
    sin_out = torch.sin(out)
    stacked = torch.stack((cos_out, -sin_out, sin_out, cos_out), dim=-1)

    # rearrange [b, n, d, 4] -> [b, n, d, 2, 2] (where 4 = 2 x 2)
    # Fast path: use contiguous reshape as 4 -> 2,2
    b, n, d, four = stacked.shape
    # Ensure the stacked dimension is always 4; if not, fallback to einops (should never happen)
    if four == 4:
        out2 = stacked.view(b, n, d, 2, 2)
    else:
        out2 = rearrange(stacked, "b n d (i j) -> b n d i j", i=2, j=2)

    return out2.to(dtype=pos_dtype, device=device)


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.view(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.view(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.view(*xq.shape).type_as(xq), xk_out.view(*xk.shape).type_as(xk)
