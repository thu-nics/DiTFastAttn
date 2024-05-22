# CUDA_VISIBLE_DEVICES=4 python main.py --n_calib 8 --n_steps 20  --window_size 128 --threshold 0.975 --eval_n_images 5000 --ablation "output_share" &
# CUDA_VISIBLE_DEVICES=5 python main.py --n_calib 8 --n_steps 20 --window_size 128 --threshold 0.95 --eval_n_images 5000 --ablation "output_share"  &
# CUDA_VISIBLE_DEVICES=6 python main.py --n_calib 8 --n_steps 20 --window_size 128 --threshold 0.925 --eval_n_images 5000 --ablation "output_share"  &
# CUDA_VISIBLE_DEVICES=7 python main.py --n_calib 8 --n_steps 20 --window_size 128 --threshold 0.9 --eval_n_images 5000 --ablation "output_share"  
# CUDA_VISIBLE_DEVICES=4 python main.py --n_calib 8 --n_steps 20 --window_size 128 --threshold 0.875 --eval_n_images 5000 --ablation "output_share"  &
# CUDA_VISIBLE_DEVICES=5 python main.py --n_calib 8 --n_steps 20 --window_size 128 --threshold 0.85 --eval_n_images 5000 --ablation "output_share" &
# CUDA_VISIBLE_DEVICES=6 python main.py --n_calib 8 --n_steps 20  --window_size 128 --threshold 0.975 --eval_n_images 5000 --ablation "full_attn+cfg_attn_share" &
# CUDA_VISIBLE_DEVICES=7 python main.py --n_calib 8 --n_steps 20 --window_size 128 --threshold 0.95 --eval_n_images 5000 --ablation "full_attn+cfg_attn_share" 
# CUDA_VISIBLE_DEVICES=4 python main.py --n_calib 8 --n_steps 20 --window_size 128 --threshold 0.925 --eval_n_images 5000 --ablation "full_attn+cfg_attn_share"  &
# CUDA_VISIBLE_DEVICES=5 python main.py --n_calib 8 --n_steps 20 --window_size 128 --threshold 0.9 --eval_n_images 5000 --ablation "full_attn+cfg_attn_share"  &
# CUDA_VISIBLE_DEVICES=6 python main.py --n_calib 8 --n_steps 20 --window_size 128 --threshold 0.875 --eval_n_images 5000 --ablation "full_attn+cfg_attn_share"  &
# CUDA_VISIBLE_DEVICES=7 python main.py --n_calib 8 --n_steps 20 --window_size 128 --threshold 0.85 --eval_n_images 5000 --ablation "full_attn+cfg_attn_share"  
# CUDA_VISIBLE_DEVICES=4 python main.py --n_calib 8 --n_steps 20  --window_size 128 --threshold 0.975 --eval_n_images 5000 --ablation "residual_window_attn" &
# CUDA_VISIBLE_DEVICES=5 python main.py --n_calib 8 --n_steps 20 --window_size 128 --threshold 0.95 --eval_n_images 5000 --ablation "residual_window_attn"  &
# CUDA_VISIBLE_DEVICES=6 python main.py --n_calib 8 --n_steps 20 --window_size 128 --threshold 0.925 --eval_n_images 5000 --ablation "residual_window_attn"  &
# CUDA_VISIBLE_DEVICES=7 python main.py --n_calib 8 --n_steps 20 --window_size 128 --threshold 0.9 --eval_n_images 5000 --ablation "residual_window_attn"  
# CUDA_VISIBLE_DEVICES=4 python main.py --n_calib 8 --n_steps 20 --window_size 128 --threshold 0.875 --eval_n_images 5000 --ablation "residual_window_attn"  &
# CUDA_VISIBLE_DEVICES=5 python main.py --n_calib 8 --n_steps 20 --window_size 128 --threshold 0.85 --eval_n_images 5000 --ablation "residual_window_attn"  &
# CUDA_VISIBLE_DEVICES=6 python main.py --n_calib 8 --n_steps 20 --window_size 128 --threshold 0.975 --eval_n_images 5000 --ablation "residual_window_attn+cfg_attn_share"  &
# CUDA_VISIBLE_DEVICES=7 python main.py --n_calib 8 --n_steps 20 --window_size 128 --threshold 0.95 --eval_n_images 5000 --ablation "residual_window_attn+cfg_attn_share"  
# CUDA_VISIBLE_DEVICES=4 python main.py --n_calib 8 --n_steps 20 --window_size 128 --threshold 0.925 --eval_n_images 5000 --ablation "residual_window_attn+cfg_attn_share"  &
# CUDA_VISIBLE_DEVICES=5 python main.py --n_calib 8 --n_steps 20 --window_size 128 --threshold 0.9 --eval_n_images 5000 --ablation "residual_window_attn+cfg_attn_share"  &
# CUDA_VISIBLE_DEVICES=6 python main.py --n_calib 8 --n_steps 20 --window_size 128 --threshold 0.875 --eval_n_images 5000 --ablation "residual_window_attn+cfg_attn_share"  &
# CUDA_VISIBLE_DEVICES=7 python main.py --n_calib 8 --n_steps 20 --window_size 128 --threshold 0.85 --eval_n_images 5000 --ablation "residual_window_attn+cfg_attn_share"



CUDA_VISIBLE_DEVICES=4 python main.py --n_calib 8 --n_steps 30  --window_size 128 --threshold 0.975 --eval_n_images 5000 &
CUDA_VISIBLE_DEVICES=5 python main.py --n_calib 8 --n_steps 30 --window_size 128 --threshold 0.95 --eval_n_images 5000  &
CUDA_VISIBLE_DEVICES=6 python main.py --n_calib 8 --n_steps 30 --window_size 128 --threshold 0.925 --eval_n_images 5000  &
CUDA_VISIBLE_DEVICES=7 python main.py --n_calib 8 --n_steps 30 --window_size 128 --threshold 0.9 --eval_n_images 5000
CUDA_VISIBLE_DEVICES=4 python main.py --n_calib 8 --n_steps 30 --window_size 128 --threshold 0.875 --eval_n_images 5000  &
CUDA_VISIBLE_DEVICES=5 python main.py --n_calib 8 --n_steps 30 --window_size 128 --threshold 0.85 --eval_n_images 5000 &
CUDA_VISIBLE_DEVICES=6 python main.py --n_calib 8 --n_steps 40  --window_size 128 --threshold 0.975 --eval_n_images 5000 &
CUDA_VISIBLE_DEVICES=7 python main.py --n_calib 8 --n_steps 40 --window_size 128 --threshold 0.95 --eval_n_images 5000
CUDA_VISIBLE_DEVICES=4 python main.py --n_calib 8 --n_steps 40 --window_size 128 --threshold 0.925 --eval_n_images 5000  &
CUDA_VISIBLE_DEVICES=5 python main.py --n_calib 8 --n_steps 40 --window_size 128 --threshold 0.9 --eval_n_images 5000  &
CUDA_VISIBLE_DEVICES=6 python main.py --n_calib 8 --n_steps 40 --window_size 128 --threshold 0.875 --eval_n_images 5000  &
CUDA_VISIBLE_DEVICES=7 python main.py --n_calib 8 --n_steps 40 --window_size 128 --threshold 0.85 --eval_n_images 5000
CUDA_VISIBLE_DEVICES=4 python main.py --n_calib 8 --n_steps 50  --window_size 128 --threshold 0.975 --eval_n_images 5000 &
CUDA_VISIBLE_DEVICES=5 python main.py --n_calib 8 --n_steps 50 --window_size 128 --threshold 0.95 --eval_n_images 5000  &
CUDA_VISIBLE_DEVICES=6 python main.py --n_calib 8 --n_steps 50 --window_size 128 --threshold 0.925 --eval_n_images 5000  &
CUDA_VISIBLE_DEVICES=7 python main.py --n_calib 8 --n_steps 50 --window_size 128 --threshold 0.9 --eval_n_images 5000
CUDA_VISIBLE_DEVICES=4 python main.py --n_calib 8 --n_steps 50 --window_size 128 --threshold 0.875 --eval_n_images 5000  &
CUDA_VISIBLE_DEVICES=5 python main.py --n_calib 8 --n_steps 50 --window_size 128 --threshold 0.85 --eval_n_images 5000 &
CUDA_VISIBLE_DEVICES=6 python main.py --n_calib 8 --n_steps 60  --window_size 128 --threshold 0.975 --eval_n_images 5000 &
CUDA_VISIBLE_DEVICES=7 python main.py --n_calib 8 --n_steps 60 --window_size 128 --threshold 0.95 --eval_n_images 5000
CUDA_VISIBLE_DEVICES=4 python main.py --n_calib 8 --n_steps 60 --window_size 128 --threshold 0.925 --eval_n_images 5000  &
CUDA_VISIBLE_DEVICES=5 python main.py --n_calib 8 --n_steps 60 --window_size 128 --threshold 0.9 --eval_n_images 5000  &
CUDA_VISIBLE_DEVICES=6 python main.py --n_calib 8 --n_steps 60 --window_size 128 --threshold 0.875 --eval_n_images 5000  &
CUDA_VISIBLE_DEVICES=7 python main.py --n_calib 8 --n_steps 60 --window_size 128 --threshold 0.85 --eval_n_images 5000


