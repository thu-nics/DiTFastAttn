CUDA_VISIBLE_DEVICES=0 python main_txt2img.py --model PixArt-alpha/PixArt-Sigma-XL-2-2K-MS --n_calib 2 --threshold 0.95 --window_size 512 --sequential --use_cache &
CUDA_VISIBLE_DEVICES=1 python main_txt2img.py --model PixArt-alpha/PixArt-Sigma-XL-2-2K-MS --n_calib 2 --threshold 0.9 --window_size 512 --sequential --use_cache &
CUDA_VISIBLE_DEVICES=2 python main_txt2img.py --model PixArt-alpha/PixArt-Sigma-XL-2-2K-MS --n_calib 2 --threshold 0.85 --window_size 512 --sequential --use_cache &
CUDA_VISIBLE_DEVICES=3 python main_txt2img.py --model PixArt-alpha/PixArt-Sigma-XL-2-2K-MS --n_calib 2 --threshold 0.8 --window_size 512 --sequential --use_cache
