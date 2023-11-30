CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 1234  train_hjr.py --cfg 'cfg_train_RA.yaml'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 1234  train_hjr.py --cfg 'cfg_train_RE.yaml'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 1234  train_hjr.py --cfg 'cfg_train_RA_drone.yaml'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 1234  train_hjr.py --cfg 'cfg_train_RE_drone.yaml'
