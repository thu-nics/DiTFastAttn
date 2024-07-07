from torchvision.transforms import functional as F
import torch.nn.functional
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
from torchmetrics.image.inception import InceptionScore
from pytorch_fid.fid_score import calculate_fid_given_paths
from torchmetrics.multimodal.clip_score import CLIPScore
from diffusers.models.attention_processor import Attention
import time


def evaluate_template_matching(order_list, cal_cost, pipe):
    count = {0: 0}
    for pattern_type in order_list:
        count[pattern_type] = 0
    mask_count_total = 0
    total = 0
    for name, module in pipe.transformer.named_modules():
        if isinstance(module, Attention):
            for timestep, mask_list in module.mask.items():
                pattern_list = np.zeros(16)
                for i in range(16):
                    type = 0
                    for j in order_list:
                        if mask_list[j][i]:
                            type = j
                            pattern_list[i] = j
                            break
                    count[type] += 1
                module.mask[timestep] = pattern_list

    total_num = sum(count.values())
    cal = 0
    for k, v in count.items():
        cal += cal_cost[k] * v / total_num
    print("template matching info: ")
    print(count)
    print("total percentage reduction: ", round(1 - cal, 2))


def preprocess_image(image):
    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2) / 255.0
    return F.center_crop(image, (256, 256))


def save_output_hook(m, i, o):
    m.saved_output = o


def test_latencies(
    pipe, n_steps, calib_x, bs, only_transformer=True, test_attention=True
):
    latencies = {}
    for b in bs:
        pipe([calib_x[0] for _ in range(b)], num_inference_steps=n_steps)
        st = time.time()
        for i in range(3):
            pipe([calib_x[0] for _ in range(b)], num_inference_steps=n_steps)
        ed = time.time()
        t = (ed - st) / 3
        if only_transformer:

            handler = pipe.transformer.register_forward_hook(save_output_hook)
            pipe([calib_x[0] for _ in range(b)], num_inference_steps=1)
            handler.remove()
            old_forward = pipe.transformer.forward
            pipe.transformer.forward = (
                lambda *arg, **kwargs: pipe.transformer.saved_output
            )
            st = time.time()
            for i in range(3):
                pipe([calib_x[0] for _ in range(b)], num_inference_steps=n_steps)
            ed = time.time()
            t_other = (ed - st) / 3
            pipe.transformer.forward = old_forward
            del pipe.transformer.saved_output
            print(f"average time for other bs={b} inference: {t_other}")
            latencies[f"{b}_other"] = t_other
            latencies[f"{b}_transformer"] = t - t_other
        print(f"average time for bs={b} inference: {t}")
        latencies[f"{b}_all"] = t

        if test_attention:
            for name, module in pipe.transformer.named_modules():
                if isinstance(module, Attention) and "attn1" in name:
                    module.old_forward = module.forward
                    module.forward = lambda *arg, **kwargs: arg[0]
            st = time.time()
            for i in range(3):
                pipe([calib_x[0] for _ in range(b)], num_inference_steps=n_steps)
            ed = time.time()
            t_other2 = (ed - st) / 3
            for name, module in pipe.transformer.named_modules():
                if isinstance(module, Attention) and "attn1" in name:
                    module.forward = module.old_forward
            t_attn = t - t_other2
            print(f"average time for attn bs={b} inference: {t_attn}")
            latencies[f"{b}_attn"] = t_attn
    return latencies


def evaluate_quantitative_scores(
    pipe,
    real_image_path,
    n_images=5000,
    batchsize=1,
    seed=3,
    num_inference_steps=20,
    fake_image_path="output/fake_images",
):
    results = {}
    # Inception Score
    inception = InceptionScore()
    # FID
    np.random.seed(seed)
    generator = torch.manual_seed(seed)
    if os.path.exists(fake_image_path):
        os.system(f"rm -rf {fake_image_path}")
    os.makedirs(fake_image_path, exist_ok=True)
    for i in range(0, n_images, batchsize):
        class_ids = np.random.randint(0, 1000, batchsize)
        output = pipe(
            class_labels=class_ids,
            generator=generator,
            output_type="np",
            num_inference_steps=num_inference_steps,
        )
        fake_images = output.images
        # Inception Score
        torch_images = (
            torch.Tensor(fake_images * 255).byte().permute(0, 3, 1, 2).contiguous()
        )
        torch_images = torch.nn.functional.interpolate(
            torch_images, size=(299, 299), mode="bilinear", align_corners=False
        )
        inception.update(torch_images)
        # save torch_image
        # for j, image in enumerate(torch_images):
        #     image = F.to_pil_image(image)
        #     image.save(f"{fake_image_path}/{i+j}_2.png")
        # save
        for j, image in enumerate(fake_images):
            image = F.to_pil_image(image)
            image.save(f"{fake_image_path}/{i+j}.png")

    IS = inception.compute()
    results["IS"] = IS
    print(f"Inception Score: {IS}")

    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    fid_value = calculate_fid_given_paths(
        [real_image_path, fake_image_path], 64, device, dims=2048, num_workers=8
    )
    results["FID"] = fid_value
    print(f"FID: {fid_value}")
    return results


def evaluate_quantitative_scores_text2img(
    pipe,
    real_image_path,
    mscoco_anno,
    n_images=5000,
    batchsize=1,
    seed=3,
    num_inference_steps=20,
    fake_image_path="output/fake_images",
    reuse_generated=True,
):
    results = {}
    # Inception Score
    inception = InceptionScore()
    clip = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    # FID
    np.random.seed(seed)
    generator = torch.manual_seed(seed)
    if os.path.exists(fake_image_path) and not reuse_generated:
        os.system(f"rm -rf {fake_image_path}")
    os.makedirs(fake_image_path, exist_ok=True)
    for index in range(0, n_images, batchsize):

        slice = mscoco_anno["annotations"][index : index + batchsize]
        filename_list = [str(d["id"]).zfill(12) for d in slice]
        # if f"{filename_list[0]}.jpg" in os.listdir(fake_image_path):
        #     continue
        print(f"Processing {index}th image")
        caption_list = [d["caption"] for d in slice]
        torch_images = []
        for filename in filename_list:
            image_file = f"{fake_image_path}/{filename}.jpg"
            if os.path.exists(image_file):
                image = Image.open(image_file)
                image_np = np.array(image)
                torch_image = torch.tensor(image_np).unsqueeze(0).permute(0, 3, 1, 2)
                torch_images.append(torch_image)
        if len(torch_images) > 0:
            torch_images = torch.cat(torch_images, dim=0)
            print(torch_images.shape)
            torch_images = torch.nn.functional.interpolate(
                torch_images, size=(256, 256), mode="bilinear", align_corners=False
            )
            inception.update(torch_images)
            clip.update(torch_images, caption_list[: len(torch_images)])
        else:
            output = pipe(
                caption_list,
                generator=generator,
                output_type="np",
                num_inference_steps=num_inference_steps,
            )
            # output = pipe(caption_list, generator = generator)
            fake_images = output.images
            # Inception Score
            count = 0
            torch_images = (
                torch.Tensor(fake_images * 255).byte().permute(0, 3, 1, 2).contiguous()
            )
            torch_images = torch.nn.functional.interpolate(
                torch_images, size=(256, 256), mode="bilinear", align_corners=False
            )
            inception.update(torch_images)
            clip.update(torch_images, caption_list)
            for j, image in enumerate(fake_images):
                # image = image.astype(np.uint8)
                image = F.to_pil_image((image * 255).astype(np.uint8))
                image.save(f"{fake_image_path}/{filename_list[count]}.jpg")
                count += 1

    IS = inception.compute()
    CLIP = clip.compute()
    results["IS"] = IS
    results["CLIP"] = CLIP
    print(f"Inception Score: {IS}")
    print(f"CLIP Score: {CLIP}")

    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    fid_value = calculate_fid_given_paths(
        [real_image_path, fake_image_path], 64, device, dims=2048, num_workers=8
    )
    results["FID"] = fid_value
    print(f"FID: {fid_value}")
    return results


def method_speed_test(pipe):
    attn = pipe.transformer.transformer_blocks[0].attn1
    all_results = []
    for seqlen in [1024, 1024 * 4, 1024 * 16]:
        print(f"Test seqlen {seqlen}")
        for method in [
            "ori",
            "full_attn",
            "full_attn+cfg_attn_share",
            "residual_window_attn",
            "residual_window_attn+cfg_attn_share",
            "output_share",
        ]:
            # for method in ["ori","full_attn","residual_window_attn","output_share"]:
            if method == "ori":
                attn.set_processor(AttnProcessor2_0())
                attn.processor.need_compute_residual = [1]
                need_compute_residuals = [False]
            else:
                attn.set_processor(FastAttnProcessor([0, 0], [method]))
                if "full_attn" in method:
                    need_compute_residuals = [False, True]
                else:
                    need_compute_residuals = [False]
            for need_compute_residual in need_compute_residuals:
                attn.processor.need_compute_residual[0] = need_compute_residual
                # warmup
                x = torch.randn(2, seqlen, attn.query_dim).half().cuda()
                for i in range(10):
                    attn.stepi = 0
                    attn(x)
                torch.cuda.synchronize()
                st = time.time()
                for i in range(1000):
                    attn.stepi = 0
                    attn(x)
                torch.cuda.synchronize()
                et = time.time()
                print(
                    f"Method {method} need_compute_residual {need_compute_residual} time {et-st}"
                )
                all_results.append(et - st)
        print(all_results)
