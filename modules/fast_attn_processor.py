import torch
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from typing import Optional
import torch.nn.functional as F
import flash_attn

# from natten.functional import na1d, na2d
import torch.nn as nn
import math


class FastAttnProcessor:
    def __init__(self, window_size, steps_method, cond_first):
        self.window_size = window_size
        self.steps_method = steps_method
        # CFG order flag (conditional first or unconditional first)
        self.cond_first = cond_first
        # Check at which timesteps do we need to compute the full-window residual
        # of this attention module
        self.need_compute_residual = self.compute_need_compute_residual(steps_method)
        self.need_cache_output = True
        self.mask_cache = {}

    def compute_need_compute_residual(self, steps_method):
        need_compute_residual = []
        for i, method in enumerate(steps_method):
            need = False
            if "full_attn" in method:
                for j in range(i + 1, len(steps_method)):
                    if "residual_window_attn" in steps_method[j] and "without_residual" not in steps_method[j]:
                        # If encountered a step that conduct WA-RS,
                        # this step needs the residual computation
                        need = True
                    if "full_attn" in steps_method[j]:
                        # If encountered another step using the `full-attn` strategy,
                        # this step doesn't need the residual computation
                        break
            need_compute_residual.append(need)
        return need_compute_residual

    def run_forward_method(self, m, hidden_states, encoder_hidden_states, attention_mask, temb, method):
        residual = hidden_states
        if method == "output_share":
            hidden_states = m.cached_output
        else:
            if "cfg_attn_share" in method:
                # Directly use the unconditional branch's attention output
                # as the conditional branch's attention output

                # TODO: Maybe use the conditional branch's attention output
                # as the unconditional's is better
                batch_size = hidden_states.shape[0]
                if self.cond_first:
                    hidden_states = hidden_states[: batch_size // 2]
                else:
                    hidden_states = hidden_states[batch_size // 2 :]
                if encoder_hidden_states is not None:
                    if self.cond_first:
                        encoder_hidden_states = encoder_hidden_states[: batch_size // 2]
                    else:
                        encoder_hidden_states = encoder_hidden_states[batch_size // 2 :]
                if attention_mask is not None:
                    if self.cond_first:
                        attention_mask = attention_mask[: batch_size // 2]
                    else:
                        attention_mask = attention_mask[batch_size // 2 :]

            if m.spatial_norm is not None:
                hidden_states = m.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            if attention_mask is not None:
                attention_mask = m.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(batch_size, m.heads, -1, attention_mask.shape[-1])

            if m.group_norm is not None:
                hidden_states = m.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = m.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif m.norm_cross:
                encoder_hidden_states = m.norm_encoder_hidden_states(encoder_hidden_states)

            key = m.to_k(encoder_hidden_states)
            value = m.to_v(encoder_hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // m.heads

            query = query.view(batch_size, -1, m.heads, head_dim)

            key = key.view(batch_size, -1, m.heads, head_dim)
            value = value.view(batch_size, -1, m.heads, head_dim)

            if attention_mask is not None:
                assert "residual_window_attn" not in method

                hidden_states = F.scaled_dot_product_attention(
                    query.transpose(1, 2),
                    key.transpose(1, 2),
                    value.transpose(1, 2),
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=False,
                ).transpose(1, 2)
            elif "full_attn" in method:
                all_hidden_states = flash_attn.flash_attn_func(query, key, value)
                if self.need_compute_residual[m.stepi]:
                    # Compute the full-window attention residual
                    w_hidden_states = flash_attn.flash_attn_func(query, key, value, window_size=self.window_size)
                    w_residual = all_hidden_states - w_hidden_states
                    if "cfg_attn_share" in method:
                        w_residual = torch.cat([w_residual, w_residual], dim=0)
                    # Save the residual for usage in follow-up steps
                    m.cached_residual = w_residual
                hidden_states = all_hidden_states
            elif "residual_window_attn" in method:
                w_hidden_states = flash_attn.flash_attn_func(query, key, value, window_size=self.window_size)

                # natten 2d
                # height = int(math.sqrt(query.shape[1]))
                # weight = height
                # q_2d = query.permute(0, 2, 1, 3).view(batch_size, m.heads, height, weight, head_dim)
                # k_2d = key.permute(0, 2, 1, 3).view(batch_size, m.heads, height, weight, head_dim)
                # v_2d = value.permute(0, 2, 1, 3).view(batch_size, m.heads, height, weight, head_dim)
                # w_hidden_states = na2d(q_2d, k_2d, v_2d, kernel_size=5)

                # natten 1d
                # w_hidden_states = na1d(query, key, value, kernel_size=self.window_size[1] * 2 - 1)

                if "without_residual" in method:
                    # For ablation study of `residual_window_attn+without_residual`
                    # v.s. `residual_window_attn`
                    hidden_states = w_hidden_states
                else:
                    hidden_states = w_hidden_states + m.cached_residual[:batch_size].view_as(w_hidden_states)

            hidden_states = hidden_states.reshape(batch_size, -1, m.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            # linear proj
            hidden_states = m.to_out[0](hidden_states)
            # dropout
            hidden_states = m.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if "cfg_attn_share" in method:
                hidden_states = torch.cat([hidden_states, hidden_states], dim=0)

            if self.need_cache_output:
                m.cached_output = hidden_states

        if m.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / m.rescale_output_factor

        return hidden_states

    def sora_attention(self, m, q, k, v, enable_flash_attn, window_size=(-1, -1)):
        if enable_flash_attn:

            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            x = flash_attn.flash_attn_func(
                q,
                k,
                v,
                dropout_p=m.attn_drop.p if m.training else 0.0,
                softmax_scale=m.scale,
                window_size=window_size,
            )
        else:
            dtype = q.dtype
            q = q * m.scale
            attn = q @ k.transpose(-2, -1)  # translate attn to float32
            attn = attn.to(torch.float32)
            if window_size != (-1, -1):
                if attn.size() not in self.mask_cache:
                    mask = (
                        torch.ones(
                            attn.size()[-2],
                            attn.size()[-1],
                            device=attn.device,
                            dtype=attn.dtype,
                        )
                        * -1e9
                    )
                    for i in range(attn.size()[-2]):
                        mask[
                            i,
                            max(0, i + window_size[0]) : min(attn.size()[-1], i + window_size[1]),
                        ] = 0
                else:
                    mask = self.mask_cache[attn.size()]
                attn += mask
            attn = attn.softmax(dim=-1)
            attn = attn.to(dtype)  # cast back attn to original dtype
            attn = m.attn_drop(attn)
            x = attn @ v
        return x

    def run_opensora_forward_method(self, m, hidden_states, encoder_hidden_states, attention_mask, temb, method):
        if method == "output_share":
            x = m.cached_output
        else:
            if "cfg_attn_share" in method:
                batch_size = hidden_states.shape[0]
                if self.cond_first:
                    diff = hidden_states[: batch_size // 2] - hidden_states[batch_size // 2 :]
                    hidden_states = hidden_states[: batch_size // 2]
                else:
                    diff = hidden_states[batch_size // 2 :] - hidden_states[: batch_size // 2]
                    hidden_states = hidden_states[batch_size // 2 :]
                # print(f"cfg_attn_share hidden_states {hidden_states.shape} max_diff={diff.abs().max()}")

                if encoder_hidden_states is not None:
                    if self.cond_first:
                        encoder_hidden_states = encoder_hidden_states[: batch_size // 2]
                    else:
                        encoder_hidden_states = encoder_hidden_states[batch_size // 2 :]
                if attention_mask is not None:
                    if self.cond_first:
                        attention_mask = attention_mask[: batch_size // 2]
                    else:
                        attention_mask = attention_mask[batch_size // 2 :]
            x = hidden_states
            B, N, C = x.shape
            # flash attn is not memory efficient for small sequences, this is empirical
            enable_flash_attn = m.enable_flash_attn and (N > B)
            qkv = m.qkv(x)
            qkv_shape = (B, N, 3, m.num_heads, m.head_dim)

            qkv = qkv.view(qkv_shape).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            # WARNING: this may be a bug
            if m.rope:
                q = m.rotary_emb(q)
                k = m.rotary_emb(k)
            q, k = m.q_norm(q), m.k_norm(k)

            if "full_attn" in method:
                x = self.sora_attention(m, q, k, v, enable_flash_attn)

                if self.need_compute_residual[m.stepi]:
                    w_x = self.sora_attention(m, q, k, v, enable_flash_attn, window_size=self.window_size)
                    # w_x=flash_attn.flash_attn_func(q, k, v,window_size=self.window_size,softmax_scale=m.scale)
                    residual = x - w_x
                    if "cfg_attn_share" in method:
                        residual = torch.cat([residual, residual], dim=0)
                    m.cached_residual = residual
            elif "residual_window_attn" in method:
                w_x = self.sora_attention(m, q, k, v, enable_flash_attn, window_size=self.window_size)
                # w_x=flash_attn.flash_attn_func(q, k, v,window_size=self.window_size,softmax_scale=m.scale)
                x = w_x + m.cached_residual[:B].view_as(w_x)

            x_output_shape = (B, N, C)
            if not enable_flash_attn:
                x = x.transpose(1, 2)
            x = x.reshape(x_output_shape)
            x = m.proj(x)
            x = m.proj_drop(x)

            if "cfg_attn_share" in method:
                x = torch.cat([x, x], dim=0)

            if self.need_cache_output:
                m.cached_output = x
        return x

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        if hasattr(attn, "qkv"):
            hidden_states = self.run_opensora_forward_method(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                temb,
                self.steps_method[attn.stepi],
            )
        else:
            hidden_states = self.run_forward_method(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                temb,
                self.steps_method[attn.stepi],
            )
        # After been call once, add the timestep index of this attention module by 1
        attn.stepi += 1
        return hidden_states
