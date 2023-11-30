import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch import optim
from torch.autograd import Variable
import network
# from param_parser import TrainParser
import argparse
import utils.utils as util
from utils.logger import create_logger
from core import test_epoch
from tensorboardX import SummaryWriter
from data.build_data import build_data_loader


import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import time
import datetime
import os
import yaml
import warnings

warnings.filterwarnings(action='ignore')
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 1234 test_hjr.py
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 --master_port 1234  test_hjr.py
#CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1234  test_hjr.py
os.chdir("/root/work/hjr/nia-hjr")

def main(cfg):
   
    log_dir, _, _ = util.make_output_directory(cfg)
    logger = create_logger(log_dir, dist_rank=0, name='')

    band =cfg['image_param']['band']
    num_class = cfg['num_class']
    #-------------------------------------------------------------------------------
    # create model
    if cfg['network']['arch'] == 'SF':
        model = network.ViT(
            image_size = cfg['network']['spectralformer']['patch'],
            near_band = cfg['network']['spectralformer']['band_patch'],
            num_patches = band,
            num_classes = num_class,
            dim = 64,
            depth = 5,
            heads = 4,
            mlp_dim = 8,
            dropout = 0.1,
            emb_dropout = 0.1,
            mode = cfg['network']['spectralformer']['mode']
        )
        # # criterion
        # criterion = nn.CrossEntropyLoss(ignore_index=30).cuda() 
        # optimizer
        opt = cfg['train_param']['opt']
        logger.info(f"optimizer = {opt} ......")
        if opt == "nadam":
            optimizer = torch.optim.NAdam(model.parameters(), lr=cfg['train_param']['learning_rate'], weight_decay=cfg['train_param']['weight_decay'])
        elif opt == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train_param']['learning_rate'], weight_decay=cfg['train_param']['weight_decay'])
        
        
        lrs = cfg['train_param']['lrs']
        logger.info(f"lr_scheduler = {lrs} ......")
        if lrs == "cosinealr":
            #create scheduler stochastic gradient descent with warm restarts(SGDR)
            tmax = np.ceil(cfg['train_param']['epoch']/cfg['train_param']['train_batch']) * 5
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= tmax, eta_min=1e-6)
        elif lrs == "steplr":
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['train_param']['epoch']//10, gamma=cfg['train_param']['gamma'])
        sample_point = util.get_point(cfg['image_param']['size'], cfg['test_param']['sampling_num'])   
    
        local_rank = int(os.environ["LOCAL_RANK"])                                            
        test_data_loader = build_data_loader(cfg['dataset'], 'test', cfg['path']['test_csv'], cfg['image_param']['type'], cfg['image_param']['isdrone'],
                                             sample_point, cfg['test_param']['test_batch'], cfg['system']['num_workers'], 
                                             local_rank, cfg['network']['spectralformer']['patch'], cfg['network']['spectralformer']['band_patch'], 
                                             cfg['image_param']['band'])     
        
    
    
    elif cfg['network']['arch'] == 'DL':
        model = network.modeling.__dict__[cfg['network']['arch']['model']](num_classes=num_class, output_stride=cfg['network']['DeepLab']['output_stride'])
        if cfg['network']['DeepLab']['separable_conv'] and 'plus' in cfg['network']['arch']['model']:
            network.convert_to_separable_conv(model.classifier)
        util.set_bn_momentum(model.backbone, momentum=0.01)
    
    
    local_rank = int(os.environ["LOCAL_RANK"])
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                              find_unused_parameters=True, broadcast_buffers=False)
    #resume
    ckpt_file_path = cfg['path']['model_path'] 
    if os.path.isfile(ckpt_file_path):
        start_epoch, best_loss, best_acc = util.load_checkpoint_files(ckpt_file_path, model, scheduler, logger)
        
    logger.info(">>>>>>>>>> Start Testing")
    start_time = time.time()
        
    
    test_res = test_epoch(model, test_data_loader, cfg, logger)       

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('ALL Test time {}'.format(total_time_str))



if __name__ == '__main__':
    
    with open('cfg_test_drone_RA.yaml') as f:
        cfg = yaml.safe_load(f)
    
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = cfg['system']['seed'] + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    
    main(cfg)



