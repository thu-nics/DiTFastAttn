CUDA_VISIBLE_DEVICES=0 python main_txt2img.py --model PixArt-alpha/PixArt-Sigma-XL-2-2K-MS --n_calib 4 --threshold 0.9 --window_size 512  --eval_n_images 500 &
CUDA_VISIBLE_DEVICES=1 python main_txt2img.py --model PixArt-alpha/PixArt-Sigma-XL-2-2K-MS --n_calib 4 --threshold 0.8 --window_size 512  --eval_n_images 500 &
CUDA_VISIBLE_DEVICES=2 python main_txt2img.py --model PixArt-alpha/PixArt-Sigma-XL-2-2K-MS --n_calib 4 --threshold 0.7 --window_size 512  --eval_n_images 500 &
CUDA_VISIBLE_DEVICES=3 python main_txt2img.py --model PixArt-alpha/PixArt-Sigma-XL-2-2K-MS --n_calib 4 --threshold 0.6 --window_size 512  --eval_n_images 500
