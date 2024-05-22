# Diffusion Transformer Fast Attention


# Install

```
conda create -n difa python=3.10
```


```
pip install torch numpy packaging matplotlib scikit-image
pip install git+https://github.com/huggingface/diffusers
pip install thop pytorch_fid torchmetrics accelerate torchmetrics[image] beautifulsoup4 ftfy flash-attn transformers SentencePiece
```

# Prepare dataset

Sample 1000 images from ImageNet to `data/real_images`.
Place coco dataset to `data/mscoco`.


# Usage
All the experiment code can be found in folder `experiments/`.

DiT compression:
```
python main.py --n_calib 8 --n_steps 20  --window_size 128 --threshold 0.975 --eval_n_images 5000 --use_cache
```

PixArt 1k compression:
```
python main_txt2img.py --n_calib 6 --n_steps 20 --window_size 512 --threshold 0.975 --eval_n_images 5000 --use_cache

```

Opensora compression:
```
python main_txt2video.py --threshold 0.975 --window_size 32 --use_cache
```
Note that before using opensora, you should install the opensora according to the readme from `https://github.com/hpcaitech/Open-Sora`. Because the opensora is under development. You should switch to v1.1 version of opensora if you meet some problem.