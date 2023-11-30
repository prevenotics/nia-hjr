CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node 2 --master_port 2434  test_hjr_RE.py
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node 2 --master_port 2434  test_hjr_RE.py
