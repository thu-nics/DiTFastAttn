import inspect
from importlib import import_module
from typing import Callable, Optional, Union
import torch
import torch.nn.functional as F
from torch import nn
from diffusers.models.attention_processor import Attention
from diffusers.utils import deprecate
attn_probs_dict={}
hidden_states_dict = {}
window_hidden_states_dict = {}
def create_local_attention_mask(height, width, window_size):
    mask = torch.zeros((height, width))
    
    for i in range(height):
        # Calculate the start and end indices for the window
        start_i = max(i - window_size // 2, 0)
        end_i = min(i + window_size // 2 + 1, height)
            
        # Set the mask for the current window
        mask[i, start_i:end_i] = 1
    
    return mask

class AttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        local_mask = create_local_attention_mask(1024, 1024, 64)
        windowed_attn_probs = attention_probs * local_mask.to(device=query.device, dtype=query.dtype)
        attn_probs_dict[self.name].append(attention_probs.detach().cpu())
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # window hidden states
        windowed_hidden_states = torch.bmm(windowed_attn_probs, value)
        windowed_hidden_states = attn.batch_to_head_dim(windowed_hidden_states)
        windowed_hidden_states = attn.to_out[0](windowed_hidden_states)
        windowed_hidden_states = attn.to_out[1](windowed_hidden_states)
        window_hidden_states_dict[self.name].append(windowed_hidden_states.detach().cpu())

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        hidden_states_dict[self.name].append(hidden_states.detach().cpu())

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states