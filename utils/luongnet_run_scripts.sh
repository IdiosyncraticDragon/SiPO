CUDA_VISIBLE_DEVICES=1 python check_model.py --model luongnet --status orig
CUDA_VISIBLE_DEVICES=1 python check_model.py --model luongnet --status pruned
