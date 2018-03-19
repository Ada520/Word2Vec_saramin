export CUDA_VISIBLE_DEVICES=2
PKL_DIR="/hd/skip-gram_minho/logs/saramin3_w4s8e128b128"

python find_topk_sim.py --pkl_dir=${PKL_DIR}
