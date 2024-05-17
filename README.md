# Diffusion Transformer Fast Attention


# Install

```
conda create -n difa python=3.10
```


```
pip install torch numpy packaging matplotlib scikit-image
pip install thop diffusers pytorch_fid torchmetrics accelerate torchmetrics[image]  flash-attn 
```

# Usage

Supported models
- facebook/DiT-XL-2-512
- PixArt-alpha/PixArt-Sigma-XL-2-2K-MS
- PixArt-alpha/PixArt-Sigma-XL-2-1024-MS