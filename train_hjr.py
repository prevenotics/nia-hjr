import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch import optim
from torch.autograd import Variable
import network
from param_parser import TrainParser
import utils.utils as util
from utils.logger import create_logger
from core import train_epoch, valid_epoch
from tensorboardX import SummaryWriter
from data.build_data import build_data_loader
from utils.tensorboard import log_tensorboard

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import time
import datetime
import os
import yaml
import warnings
import argparse


warnings.filterwarnings(action='ignore')

#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 --master_port 1234  train_hjr.py


def main(cfg):

    start_epoch=0
    num_epochs=cfg['train_param']['epoch']
    best_loss = 9999
    best_acc = -9999
    
    
    log_dir, pth_dir, tensorb_dir = util.make_output_directory(cfg)
    logger = create_logger(log_dir, dist_rank=0, name='')
    
    
    
    #create tensorboard
    if cfg['system']['tensorboard']:
        writer_tb = SummaryWriter(log_dir=tensorb_dir)

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
        # criterion
        criterion = nn.CrossEntropyLoss(ignore_index=30).cuda() 
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
        sample_point = util.get_point(cfg['image_param']['size'], cfg['network']['spectralformer']['sampling_num'])   
    
        local_rank = int(os.environ["LOCAL_RANK"])                                            
        train_data_loader = build_data_loader(cfg['dataset'], 'train', cfg['path']['train_csv'], cfg['image_param']['type'], cfg['image_param']['isdrone'], sample_point, cfg['train_param']['train_batch'], cfg['system']['num_workers'], local_rank, cfg['network']['spectralformer']['patch'], cfg['network']['spectralformer']['band_patch'], cfg['image_param']['band'])     
        val_data_loader   = build_data_loader(cfg['dataset'], 'train', cfg['path']['val_csv'],   cfg['image_param']['type'], cfg['image_param']['isdrone'], sample_point, cfg['train_param']['val_batch'],   cfg['system']['num_workers'], local_rank, cfg['network']['spectralformer']['patch'], cfg['network']['spectralformer']['band_patch'], cfg['image_param']['band'])
   

    
    local_rank = int(os.environ["LOCAL_RANK"])
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                              find_unused_parameters=True, broadcast_buffers=False)
    #resume
    ckpt_file_path = os.path.join(pth_dir, "checkpoint.pth")
    if cfg['train_param']['resume'] and os.path.isfile(ckpt_file_path):
        start_epoch, best_loss, best_acc = util.load_checkpoint_files(ckpt_file_path, model, scheduler, logger)
        
    
    best_loss_ckpt_file_path = os.path.join(pth_dir, "best_checkpoint_loss.pth")
    best_OA_ckpt_file_path = os.path.join(pth_dir, "best_checkpoint_OA.pth")
    

    
    logger.info(">>>>>>>>>> Start training")
    start_time = time.time()
    
    tar = np.array([])
    pre = np.array([])
        
    for epoch in range(start_epoch, cfg['train_param']['epoch']): 
        
        scheduler.step()

        model.train()        
        train_res = train_epoch(model, tar, pre, train_data_loader, criterion, optimizer, scheduler, epoch, cfg, logger)
        print("Train Epoch: {:03d} train_acc: {:.2f} train_loss: {:.4f} train_OA: {:.4f} train_Kappa: {:.4f}".
              format(epoch+1, train_res[0].avg, train_res[1].avg, train_res[2].avg, train_res[3].avg))
            
        if dist.get_rank() == 0:
            util.save_checkpoint(model, optimizer, scheduler, epoch, ckpt_file_path, best_loss, best_acc)
            logger.info(f">>>>> {ckpt_file_path}saved......")
            if epoch % (cfg['train_param']['save_freq'])==0 or epoch==num_epochs-1:
                save_path = os.path.join(pth_dir, f"checkpoint_%06d.pth"%epoch)
                util.save_checkpoint(model, optimizer, scheduler, epoch, save_path, best_loss, best_acc)
                logger.info(f">>>>> {save_path}saved......")

        if cfg['system']['tensorboard']:
            current_log = {'TRAIN_0_acc': train_res[0].avg,
                           'TRAIN_1_loss': train_res[1].avg,
                           'TRAIN_2_OA': train_res[2].avg,
                           'TRAIN_3_Kappa': train_res[3].avg,                                              
                           'TRAIN_4_lr_avg': train_res[4].avg}
            log_tensorboard(writer_tb, current_log, epoch)
            
            
        # evaluation
        if (epoch % cfg['train_param']['eval_freq'] == 0) or (epoch == num_epochs - 1):        
            val_res = valid_epoch(model, val_data_loader, criterion, epoch, cfg, logger)            
            print("Validation Epoch: {:03d} val_acc: {:.2f} val_loss: {:.4f} val_OA: {:.4f} val_Kappa: {:.4f}".
              format(epoch+1, val_res[0].avg, val_res[1].avg, val_res[2].avg, val_res[3].avg))
            
            if cfg['system']['tensorboard']:
                current_log = {'VAL_0_acc': val_res[0].avg,
                            'VAL_1_loss': val_res[1].avg,
                            'VAL_2_OA': val_res[2].avg,
                            'VAL_3_Kappa': val_res[3].avg}
                log_tensorboard(writer_tb, current_log, epoch)

            val_acc = val_res[2].avg
            val_loss = val_res[1].avg
            
            if dist.get_rank() == 0:
                if val_loss < best_loss:
                    best_loss = val_loss
                    # save ckpt
                    util.save_checkpoint(model, optimizer, scheduler, epoch, best_loss_ckpt_file_path, best_loss=best_loss, best_acc=val_acc)                    
                    logger.info(f">>>>> best loss {best_loss_ckpt_file_path}_{epoch}_{best_loss} saved......")

                if val_acc > best_acc:
                    best_acc = val_acc
                    # save ckpt
                    util.save_checkpoint(model, optimizer, scheduler, epoch, best_OA_ckpt_file_path, best_loss=val_loss, best_acc=best_acc)
                    logger.info(f">>>>> best acc {best_OA_ckpt_file_path}_{epoch}_{best_acc} saved......")    

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('ALL Training time {}'.format(total_time_str))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--cfg', help='cgf_train_LU_RE.yaml')
    parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    args = parser.parse_args()

    with open(args.cfg) as f:
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



