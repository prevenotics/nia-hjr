UDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 1234  train_hjr_drone_RE.py
UDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 1234  train_hjr_drone_RA.py
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 --master_port 2234  test_hjr_drone_RE.py
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node 2 --master_port 2234  test_hjr_drone_RA.py
