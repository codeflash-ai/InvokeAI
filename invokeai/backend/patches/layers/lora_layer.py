from typing import Dict, Optional

import torch

from invokeai.backend.patches.layers.lora_layer_base import LoRALayerBase
from invokeai.backend.util.calc_tensor_size import calc_tensors_size


class LoRALayer(LoRALayerBase):
    def __init__(
        self,
        up: torch.Tensor,
        mid: Optional[torch.Tensor],
        down: torch.Tensor,
        alpha: float | None,
        bias: Optional[torch.Tensor],
    ):
        super().__init__(alpha, bias)
        self.up = up
        self.mid = mid
        self.down = down
        self.are_ranks_equal = up.shape[1] == down.shape[0]

    @classmethod
    def from_state_dict_values(
        cls,
        values: Dict[str, torch.Tensor],
    ):
        alpha = cls._parse_alpha(values.get("alpha", None))
        bias = cls._parse_bias(
            values.get("bias_indices", None), values.get("bias_values", None), values.get("bias_size", None)
        )

        layer = cls(
            up=values["lora_up.weight"],
            down=values["lora_down.weight"],
            mid=values.get("lora_mid.weight", None),
            alpha=alpha,
            bias=bias,
        )

        cls.warn_on_unhandled_keys(
            values=values,
            handled_keys={
                # Default keys.
                "alpha",
                "bias_indices",
                "bias_values",
                "bias_size",
                # Layer-specific keys.
                "lora_up.weight",
                "lora_down.weight",
                "lora_mid.weight",
            },
        )

        return layer

    def _rank(self) -> int:
        return self.down.shape[0]

    def fuse_weights(self, up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
        """
        Fuse the weights of the up and down matrices of a LoRA layer with different ranks.

        Since the Huggingface implementation of KQV projections are fused, when we convert to Kohya format
        the LoRA weights have different ranks. This function handles the fusion of these differently sized
        matrices.
        """
        up_rows, up_cols = up.shape
        down_rows, down_cols = down.shape
        fused_lora = torch.zeros((up_rows, down_cols), device=down.device, dtype=down.dtype)
        rank_diff = down_rows / up_cols

        # optimization: coalesce operation with torch.stack and .sum to avoid python-level looping

        if rank_diff > 1:
            # split down along the row dimension and compute mm for each chunk
            num_chunks = int(rank_diff)
            w_down = down.chunk(num_chunks, dim=0)
            # if num_chunks == 1, fallback to simpler matmul
            if num_chunks == 1:
                fused_lora = torch.mm(up, down)
            else:
                fused_lora = torch.stack([torch.mm(up, w) for w in w_down], dim=0).sum(dim=0)
        else:
            rank_diff = up_cols / down_rows
            num_chunks = int(rank_diff)
            w_up = up.chunk(num_chunks, dim=0)
            if num_chunks == 1:
                fused_lora = torch.mm(up, down)
            else:
                fused_lora = torch.stack([torch.mm(w, down) for w in w_up], dim=0).sum(dim=0)

        return fused_lora

    def get_weight(self, orig_weight: torch.Tensor) -> torch.Tensor:
        if self.mid is not None:
            up = self.up.reshape(self.up.shape[0], self.up.shape[1])
            down = self.down.reshape(self.down.shape[0], self.down.shape[1])
            weight = torch.einsum("m n w h, i m, n j -> i j w h", self.mid, up, down)
        else:
            # up matrix and down matrix have different ranks so we can't simply multiply them
            if not self.are_ranks_equal:
                weight = self.fuse_weights(self.up, self.down)
                return weight

            weight = self.up.reshape(self.up.shape[0], -1) @ self.down.reshape(self.down.shape[0], -1)

        return weight

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().to(device=device, dtype=dtype)
        self.up = self.up.to(device=device, dtype=dtype)
        if self.mid is not None:
            self.mid = self.mid.to(device=device, dtype=dtype)
        self.down = self.down.to(device=device, dtype=dtype)

    def calc_size(self) -> int:
        return super().calc_size() + calc_tensors_size([self.up, self.mid, self.down])
