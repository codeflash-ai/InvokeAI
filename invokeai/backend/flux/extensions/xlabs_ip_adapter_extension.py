import math
from typing import List, Union

import einops
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from invokeai.backend.flux.ip_adapter.xlabs_ip_adapter_flux import XlabsIpAdapterFlux
from invokeai.backend.flux.modules.layers import DoubleStreamBlock
from invokeai.backend.util.devices import TorchDevice


class XLabsIPAdapterExtension:
    def __init__(
        self,
        model: XlabsIpAdapterFlux,
        image_prompt_clip_embed: torch.Tensor,
        weight: Union[float, List[float]],
        begin_step_percent: float,
        end_step_percent: float,
    ):
        self._model = model
        self._image_prompt_clip_embed = image_prompt_clip_embed
        self._weight = weight
        self._begin_step_percent = begin_step_percent
        self._end_step_percent = end_step_percent

        self._image_proj: torch.Tensor | None = None

        # Precompute to optimize repetitive calls to _get_weight with same step counts
        self._last_total_num_timesteps: int | None = None
        self._first_step: int | None = None
        self._last_step: int | None = None

    def _get_weight(self, timestep_index: int, total_num_timesteps: int) -> float:
        # Avoid recomputation if step count is unchanged
        if self._last_total_num_timesteps != total_num_timesteps:
            self._first_step = int(self._begin_step_percent * total_num_timesteps)
            # math.floor can be used directly on multiplication result for int
            self._last_step = int(math.ceil(self._end_step_percent * total_num_timesteps))
            self._last_total_num_timesteps = total_num_timesteps

        first_step = self._first_step
        last_step = self._last_step

        if timestep_index < first_step or timestep_index > last_step:
            return 0.0

        weight = self._weight
        # Reorder type check to avoid isinstance for every possible call
        if isinstance(weight, list):
            return weight[timestep_index]

        return weight

    @staticmethod
    def run_clip_image_encoder(
        pil_image: List[Image.Image], image_encoder: CLIPVisionModelWithProjection
    ) -> torch.Tensor:
        clip_image_processor = CLIPImageProcessor()
        clip_image: torch.Tensor = clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(device=TorchDevice.choose_torch_device(), dtype=image_encoder.dtype)
        clip_image_embeds = image_encoder(clip_image).image_embeds
        return clip_image_embeds

    def run_image_proj(self, dtype: torch.dtype):
        image_prompt_clip_embed = self._image_prompt_clip_embed.to(dtype=dtype)
        self._image_proj = self._model.image_proj(image_prompt_clip_embed)

    def run_ip_adapter(
        self,
        timestep_index: int,
        total_num_timesteps: int,
        block_index: int,
        block: DoubleStreamBlock,
        img_q: torch.Tensor,
        img: torch.Tensor,
    ) -> torch.Tensor:
        """The logic in this function is based on:
        https://github.com/XLabs-AI/x-flux/blob/47495425dbed499be1e8e5a6e52628b07349cba2/src/flux/modules/layers.py#L245-L301
        """
        weight = self._get_weight(timestep_index=timestep_index, total_num_timesteps=total_num_timesteps)
        if weight < 1e-6:
            return img

        ip_adapter_block = self._model.ip_adapter_double_blocks.double_blocks[block_index]

        ip_key = ip_adapter_block.ip_adapter_double_stream_k_proj(self._image_proj)
        ip_value = ip_adapter_block.ip_adapter_double_stream_v_proj(self._image_proj)

        # Reshape projections for multi-head attention.
        ip_key = einops.rearrange(ip_key, "B L (H D) -> B H L D", H=block.num_heads)
        ip_value = einops.rearrange(ip_value, "B L (H D) -> B H L D", H=block.num_heads)

        # Compute attention between IP projections and the latent query.
        ip_attn = torch.nn.functional.scaled_dot_product_attention(
            img_q, ip_key, ip_value, dropout_p=0.0, is_causal=False
        )
        ip_attn = einops.rearrange(ip_attn, "B H L D -> B L (H D)", H=block.num_heads)

        img = img + weight * ip_attn

        return img
