import torch
# import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from scipy.io import loadmat
from scipy.io import savemat
from torch import optim
from torch.autograd import Variable
# from model.vit_pytorch import ViT # For SpectralFormer
import network
from param_parser import TrainParser
# import utils
from utils.utils import make_output_directory, load_checkpoint_files, save_checkpoint, chooose_train_and_test_point, mirror_hsi, gain_neighborhood_pixel, gain_neighborhood_band, train_and_test_data, train_and_test_label, accuracy, output_metric, cal_results, get_point, set_bn_momentum
from utils.logger import create_logger
from core import train_epoch, valid_epoch, test_epoch
from tensorboardX import SummaryWriter
from data.hjr_dataset import HJRDataset
from data.build_data import build_data_loader
from utils.tensorboard import log_tensorboard

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import time
import datetime
import os
import yaml

# import sys
# sys.argv=['']
# del sys
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 1234 train_hjr.py --eval_freq 1
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 --master_port 1234  train_hjr.py --eval_freq 1 --save_freq 100
os.chdir("/root/work/hjr/IEEE_TGRS_SpectralFormer")

def main(cfg):

    # # Parameter Setting
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # cudnn.deterministic = True
    # cudnn.benchmark = False
    start_epoch=0
    # num_epochs=args.epoches
    num_epochs=cfg['train_param']['epoch']
    best_loss = 9999
    best_acc = -9999
    
    
    log_dir, pth_dir, tensorb_dir = make_output_directory(cfg)
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
        criterion = nn.CrossEntropyLoss(ignore_index=255).cuda() 
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
        sample_point = get_point(cfg['image_param']['size'], cfg['network']['spectralformer']['sampling_num'])   
    
        local_rank = int(os.environ["LOCAL_RANK"])                                            
        test_data_loader = build_data_loader(cfg['dataset'], cfg['path']['test_csv'], cfg['image_param']['type'], sample_point, cfg['test_param']['test_batch'], cfg['system']['num_workers'], local_rank, cfg['network']['spectralformer']['patch'], cfg['network']['spectralformer']['band_patch'], cfg['image_param']['band'])     
        
    
    
    elif cfg['network']['arch'] == 'DL':
        model = network.modeling.__dict__[cfg['network']['arch']['model']](num_classes=num_class, output_stride=cfg['network']['DeepLab']['output_stride'])
        if cfg['network']['DeepLab']['separable_conv'] and 'plus' in cfg['network']['arch']['model']:
            network.convert_to_separable_conv(model.classifier)
        set_bn_momentum(model.backbone, momentum=0.01)
    
        
    
    
    
    # train_data_loader = build_data_loader(args.dataset, train_csv_path, imgtype, sample_point, args.train_batch_size, args.num_workers, args.local_rank, args.patch, args.band_patch, args.band)    
    # val_data_loader   = build_data_loader(args.dataset, val_csv_path,   imgtype, sample_point, args.val_batch_size,   args.num_workers, args.local_rank, args.patch, args.band_patch, args.band)
    # train_data_loader=Data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
    


    
    local_rank = int(os.environ["LOCAL_RANK"])
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                              find_unused_parameters=True, broadcast_buffers=False)
    #resume
    ckpt_file_path = cfg['path']['model_path'] #os.path.join(pth_dir, "checkpoints.pth")
    if os.path.isfile(ckpt_file_path):
        start_epoch, best_loss, best_acc = load_checkpoint_files(ckpt_file_path, model, scheduler, logger)
        
    
    # best_loss_ckpt_file_path = os.path.join(pth_dir, "best_checkpoints_loss.pth")
    # best_acc_c_ckpt_file_path = os.path.join(pth_dir, "best_checkpoints_acc_corr.pth")
    # best_acc_r_ckpt_file_path = os.path.join(pth_dir, "best_checkpoints_acc_r.pth")

    
    logger.info(">>>>>>>>>> Start Testing")
    start_time = time.time()
        
    
    test_res = test_epoch(model, test_data_loader, cfg, logger)       

            
            
            

        # if (epoch % args.test_freq == 0) | (epoch == args.epoches - 1):         
        #     model.eval()
        #     tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
        #     OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
        #     print("Epoch: {:03d} val_OA: {:.4f}val_Kappa: {:.4f}".format(epoch+1, OA2, Kappa2))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('ALL Training time {}'.format(total_time_str))



if __name__ == '__main__':
    
    

    # parser = argparse.ArgumentParser("HSI")
    # parser.add_argument('--dataset', choices=['Indian', 'Pavia', 'Houston'], default='Indian', help='dataset to use')
    # parser.add_argument('--flag_test', choices=['test', 'train'], default='train', help='testing mark')
    
    # parser = TrainParser()
    # args = parser.parse_args(args=[])
    
    with open('./config.yaml') as f:
        cfg = yaml.safe_load(f)
        

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    # torch.cuda.set_device(args.local_rank)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = cfg['system']['seed'] + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    
    main(cfg)



