import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from diffusers.models.transformers.transformer_2d import Transformer2DModel
import functools
import collections
from modules.fast_feed_forward import FastFeedForward
from modules.fast_attn_processor import FastAttnProcessor
import os
from time import time

from diffusers.models import AutoencoderKL


def set_stepi_warp(pipe):
    @functools.wraps(pipe)
    def wrapper(*args, **kwargs):
        for blocki, block in enumerate(pipe.transformer.transformer_blocks):
            for layer in block.children():
                layer.stepi = 0
                layer.cached_residual = None
                layer.cached_output = None
        out = pipe(*args, **kwargs)

        for blocki, block in enumerate(pipe.transformer.transformer_blocks):
            for layer in block.children():
                layer.stepi = 0
                layer.cached_residual = None
                layer.cached_output = None
        return out

    return wrapper


def compression_loss(a, b, metric=""):
    ls = []
    if a.__class__.__name__ == "Transformer2DModelOutput":
        a = [a.sample]
        b = [b.sample]
    for ai, bi in zip(a, b):
        if isinstance(ai, torch.Tensor):
            if metric == "ssim":
                ssim = SSIM(data_range=1.0).to(ai.device)
                l = 1 - ssim(ai, bi)
            elif metric == "lpips":
                vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device=ai.device, dtype=ai.dtype)
                lpips = LPIPS(net_type="squeeze").to(ai.device)
                l = lpips(
                    vae.decode(ai.reshape(ai.shape[0] * 2, ai.shape[1] // 2, ai.shape[2], ai.shape[3])).sample,
                    vae.decode(bi.reshape(ai.shape[0] * 2, ai.shape[1] // 2, ai.shape[2], ai.shape[3])).sample,
                )
            else:
                diff = (ai - bi) / (torch.max(ai, bi) + 1e-6)
                l = diff.abs().clip(0, 10).mean()
            ls.append(l)
    l = sum(ls) / len(ls)
    return l


def transformer_forward_pre_hook(m, args, kwargs):
    now_stepi = m.transformer_blocks[0].attn1.stepi
    for blocki, block in enumerate(m.transformer_blocks):
        # Set `need_compute_residual` to False to avoid the process of trying different
        # compression strategies to override the saved residual.
        block.attn1.processor.need_compute_residual[now_stepi] = False
        block.attn1.processor.need_cache_output = False
    raw_outs = m.forward(*args, **kwargs)
    for blocki, block in enumerate(m.transformer_blocks):
        if now_stepi == 0:
            continue
        # Currently, we only compress `attn1` in each block. `attn2` is not handled.
        for attni, attn in enumerate([block.attn1]):
            if attn is None or not isinstance(attn.processor, FastAttnProcessor):
                continue
            method_candidates = block.method_candidates
            selected_method = "full_attn"
            for method in method_candidates:
                # Try compress this attention using `method`
                attn.processor.steps_method[now_stepi] = method

                # Set the timestep index of every layer back to now_stepi
                # (which are increased by one in every forward)
                for _block in m.transformer_blocks:
                    for layer in _block.children():
                        layer.stepi = now_stepi

                # Compute the overall transformer output
                outs = m.forward(*args, **kwargs)

                l = compression_loss(raw_outs, outs, metric=m.metric)
                threshold = m.loss_thresholds[now_stepi][blocki]

                if m.debug:
                    print(f"{method}: L(O,O')={l} threshold={threshold}")
                if l < threshold:
                    selected_method = method
                    break

            attn.processor.steps_method[now_stepi] = selected_method
            print(f"Block {blocki} attn{attni} stepi{now_stepi} {selected_method}")
            del l, outs
    del raw_outs

    # Set the timestep index of every layer back to now_stepi
    # (which are increased by one in every forward)
    for _block in m.transformer_blocks:
        for layer in _block.children():
            layer.stepi = now_stepi

    for blocki, block in enumerate(m.transformer_blocks):
        # During the compression plan decision process,
        # we set the `need_compute_residual` property of all attention modules to `True`,
        # so that all full attention modules will save its residual for convenience.
        # The residual will be saved in the follow-up forward call.
        block.attn1.processor.need_compute_residual[now_stepi] = True
        block.attn1.processor.need_cache_output = True


@torch.no_grad()
def transform_model_fast_attention(
    raw_pipe,
    n_steps,
    n_calib,
    calib_x,
    threshold,
    window_size=[-64, 64],
    use_cache=False,
    seed=3,
    sequential_calib=False,
    debug=False,
    ablation="",
    cond_first=False,
    metric="",
    negative_prompt="",
    guidance_scale=4,
):
    pipe = set_stepi_warp(raw_pipe)
    blocks = pipe.transformer.transformer_blocks
    transformer = pipe.transformer
    # is_transform_attn2=blocks[0].attn2 is not None
    is_transform_attn2 = False
    print(f"Transform attn2 {is_transform_attn2}")
    # is_transform_ff=hasattr(blocks[0],"ff")
    is_transform_ff = False
    print(f"Transform ff {is_transform_ff}")

    st = time()
    cache_file = f"cache/{raw_pipe.config._name_or_path.replace('/','_')}_{n_steps}_{n_calib}_{threshold}_{sequential_calib}_{window_size}_{guidance_scale}"
    if ablation != "":
        cache_file = cache_file + f"_{ablation}"
    if metric != "":
        cache_file = cache_file + f"_{metric}"
    if negative_prompt != "":
        cache_file = cache_file + f"_{negative_prompt}"

    cache_file = cache_file + ".json"
    print(f"cache file is {cache_file}")
    if use_cache and os.path.exists(cache_file):
        blocks_methods = torch.load(cache_file)
    else:
        # reset all processors
        for blocki, block in enumerate(blocks):
            attn: Attention = block.attn1
            if ablation != "":
                block.method_candidates = ablation.split(",") if isinstance(ablation, str) else ablation
            else:
                block.method_candidates = [
                    "output_share",  # AST
                    "residual_window_attn+cfg_attn_share",  # WA-RS + ASC
                    "residual_window_attn",  # WA-RS
                    "full_attn+cfg_attn_share",  # ASC
                ]
            print(f"method_candidates of {blocki} {block.method_candidates}")

            # Initialize all attention processors to the `full_attn` strategy
            block.attn1.processor = FastAttnProcessor(
                window_size, ["full_attn" for _ in range(n_steps)], cond_first=cond_first
            )
            block.attn1.processor.need_compute_residual = [True for _ in range(n_steps)]
            if is_transform_attn2:
                block.attn2.processor = FastAttnProcessor(
                    window_size, ["full_attn" for _ in range(n_steps)], cond_first=cond_first
                )
                block.attn2.processor.need_compute_residual = [True for _ in range(n_steps)]
            if is_transform_ff:
                block.ff = FastFeedForward(block.ff.net, ["full_attn" for _ in range(n_steps)])

        # Setup loss threshold for each timestep and layer
        loss_thresholds = []
        for step_i in range(n_steps):
            sub_list = []
            for blocki in range(len(blocks)):
                threshold_i = (blocki + 1) / len(blocks) * threshold
                sub_list.append(threshold_i)
            loss_thresholds.append(sub_list)

        # calibration
        print(isinstance(transformer, Transformer2DModel))
        transformer.metric = metric
        h = transformer.register_forward_pre_hook(transformer_forward_pre_hook, with_kwargs=True)
        ##########
        def test_hook(m, input, output):
        # output[0].register_hook()
            print("forward done ------- -------")
        ##########
        h_test = transformer.register_forward_hook(test_hook)
        transformer.loss_thresholds = loss_thresholds
        transformer.pipe = pipe
        transformer.debug = debug
        print(transformer)

        if negative_prompt == "":
            pipe(
                calib_x,
                num_inference_steps=n_steps,
                generator=torch.manual_seed(seed),
                output_type="latent",
                return_dict=False,
                guidance_scale=guidance_scale,
            )
        else:
            pipe(
                calib_x,
                num_inference_steps=n_steps,
                generator=torch.manual_seed(seed),
                output_type="latent",
                negative_prompt=negative_prompt,
                return_dict=False,
                guidance_scale=guidance_scale,
            )

        h.remove()
        h_test.remove()

        blocks_methods = []
        for blocki, block in enumerate(blocks):
            attn_steps_method = block.attn1.processor.steps_method
            attn2_steps_method = block.attn2.processor.steps_method if is_transform_attn2 else None
            ff_steps_method = block.ff.steps_method if is_transform_ff else None
            blocks_methods.append(
                {
                    "attn1": attn_steps_method,
                    "attn2": attn2_steps_method,
                    "ff": ff_steps_method,
                }
            )

        # save cache
        if not os.path.exists("cache"):
            os.makedirs("cache")
        torch.save(blocks_methods, cache_file)

    et = time()

    # set processor
    for blocki, block in enumerate(blocks):
        block.attn1.processor = FastAttnProcessor(window_size, blocks_methods[blocki]["attn1"], cond_first=cond_first)
        if blocks_methods[blocki]["attn2"] is not None:
            block.attn2.processor = FastAttnProcessor(
                window_size, blocks_methods[blocki]["attn2"], cond_first=cond_first
            )
        if blocks_methods[blocki]["ff"] is not None:
            block.ff = FastFeedForward(block.ff.net, blocks_methods[blocki]["ff"])

    # statistics
    counts = collections.Counter([method for block in blocks for method in block.attn1.processor.steps_method])
    total = sum(counts.values())
    for k, v in counts.items():
        print(f"attn1 {k} {v/total}")

    if is_transform_attn2:
        counts = collections.Counter([method for block in blocks for method in block.attn2.processor.steps_method])
        total = sum(counts.values())
        for k, v in counts.items():
            print(f"attn2 {k} {v/total}")
    if is_transform_ff:
        counts = collections.Counter([method for block in blocks for method in block.ff.steps_method])
        total = sum(counts.values())
        for k, v in counts.items():
            print(f"ff {k} {v/total}")

    return pipe, et - st
