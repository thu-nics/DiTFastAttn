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

# Usage

Supported models
- facebook/DiT-XL-2-512
- PixArt-alpha/PixArt-Sigma-XL-2-2K-MS
- PixArt-alpha/PixArt-Sigma-XL-2-1024-MS