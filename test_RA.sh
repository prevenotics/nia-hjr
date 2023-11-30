CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 --master_port 1234  test_hjr.py --cfg 'cfg_test_RA.yaml'
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 --master_port 1234  test_hjr.py --cfg 'cfg_test_RE.yaml'
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 --master_port 1234  test_hjr.py --cfg 'cfg_test_RA_drone.yaml'
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 --master_port 1234  test_hjr.py --cfg 'cfg_test_RE_drone.yaml'
