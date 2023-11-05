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
from utils.utils import make_output_directory, load_checkpoint_files, save_checkpoint, chooose_train_and_test_point, mirror_hsi, gain_neighborhood_pixel, gain_neighborhood_band, train_and_test_data, train_and_test_label, accuracy, output_metric, cal_results, get_point
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
        train_data_loader = build_data_loader(cfg['dataset'], cfg['path']['train_csv'], cfg['image_param']['type'], sample_point, cfg['train_param']['train_batch'], cfg['system']['num_workers'], local_rank, cfg['network']['spectralformer']['patch'], cfg['network']['spectralformer']['band_patch'], cfg['image_param']['band'])     
        val_data_loader   = build_data_loader(cfg['dataset'], cfg['path']['val_csv'],   cfg['image_param']['type'], sample_point, cfg['train_param']['val_batch'],   cfg['system']['num_workers'], local_rank, cfg['network']['spectralformer']['patch'], cfg['network']['spectralformer']['band_patch'], cfg['image_param']['band'])
    
    
    elif cfg['network']['arch'] == 'DL':
        model = network.modeling.__dict__[cfg['network']['arch']['model']](num_classes=num_class, output_stride=cfg['network']['DeepLab']['output_stride'])
        if opts.separable_conv and 'plus' in cfg['network']['arch']['model']:
            network.convert_to_separable_conv(model.classifier)
        utils.set_bn_momentum(model.backbone, momentum=0.01)
    
        
    
    
    
    # train_data_loader = build_data_loader(args.dataset, train_csv_path, imgtype, sample_point, args.train_batch_size, args.num_workers, args.local_rank, args.patch, args.band_patch, args.band)    
    # val_data_loader   = build_data_loader(args.dataset, val_csv_path,   imgtype, sample_point, args.val_batch_size,   args.num_workers, args.local_rank, args.patch, args.band_patch, args.band)
    # train_data_loader=Data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
    


    
    local_rank = int(os.environ["LOCAL_RANK"])
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                              find_unused_parameters=True, broadcast_buffers=False)
    #resume
    ckpt_file_path = os.path.join(pth_dir, "checkpoints.pth")
    if cfg['train_param']['resume'] and os.path.isfile(ckpt_file_path):
        start_epoch, best_loss, best_acc = load_checkpoint_files(ckpt_file_path, model, scheduler, logger)
        
    
    best_loss_ckpt_file_path = os.path.join(pth_dir, "best_checkpoints_loss.pth")
    best_acc_c_ckpt_file_path = os.path.join(pth_dir, "best_checkpoints_acc_corr.pth")
    best_acc_r_ckpt_file_path = os.path.join(pth_dir, "best_checkpoints_acc_r.pth")

    
    logger.info(">>>>>>>>>> Start training")
    start_time = time.time()
    
    tar = np.array([])
    pre = np.array([])
        
    for epoch in range(cfg['train_param']['epoch']): 
        
        scheduler.step()

        # train model
        model.train()
        # train_acc, train_obj, tar_t, pre_t = train_epoch(model, train_data_loader, criterion, optimizer)
              # def train_epoch(model, train_loader, criterion, optimizer, lr_scheduler, epoch, num_epochs, logger, print_freq=1000):
        train_res = train_epoch(model, tar, pre, train_data_loader, criterion, optimizer, scheduler, epoch, cfg, logger)
        # OA1, AA_mean1, Kappa1, AA1 = output_metric(tar_t, pre_t) 
        print("Epoch: {:03d} train_acc: {:.4f} train_loss: {:.4f} train_OA: {:.4f} train_Kappa: {:.4f}".
              format(epoch+1, train_res[0].avg, train_res[1].avg, train_res[2].avg, train_res[3].avg))
        
        if dist.get_rank() == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, ckpt_file_path, best_loss, best_acc)
            logger.info(f">>>>> {ckpt_file_path}saved......")
            if epoch % (cfg['train_param']['save_freq'])==0 or epoch==num_epochs-1:
                save_path = os.path.join(pth_dir, f"checkpoint_%06d.pth"%epoch)
                save_checkpoint(model, optimizer, scheduler, epoch, save_path, best_loss, best_acc)
                logger.info(f">>>>> {save_path}saved......")

        if cfg['system']['tensorboard']:
            current_log = {'TRAIN_0_acc': train_res[0].avg,
                           'TRAIN_1_loss': train_res[1].avg,
                           'TRAIN_2_OA': train_res[2].avg,
                           'TRAIN_3_Kappa': train_res[3].avg,                                              
                           'TRAIN_4_lr_avg': train_res[4].avg}
            log_tensorboard(writer_tb, current_log, epoch)
            
            
        # evaluation
        # if (epoch % args.eval_freq == 0) or (epoch == num_epochs - 1):        
        if cfg['train_param']['eval_freq'] == 1:        
            val_res = valid_epoch(model, val_data_loader, criterion, epoch, cfg, logger)            

            if cfg['system']['tensorboard']:
                current_log = {'VAL_0_acc': val_res[0].avg,
                            'VAL_1_loss': val_res[1].avg,
                            'VAL_2_OA': val_res[2].avg,
                            'VAL_3_Kappa': val_res[3].avg}
                log_tensorboard(writer_tb, current_log, epoch)
            

            val_acc = val_res[2].avg
            val_loss = val_res[1].avg
            
            # val_acc_r = val_res[4].avg
            if dist.get_rank() == 0:
                if val_loss < best_loss:
                    best_loss = val_loss
                    # save ckpt
                    save_checkpoint(model, optimizer, scheduler, epoch, best_loss_ckpt_file_path, best_loss=best_loss, best_acc=val_acc)                    
                    logger.info(f">>>>> best loss {best_loss_ckpt_file_path}_{epoch}_{best_loss} saved......")

                if val_acc > best_acc:
                    best_acc = val_acc
                    # save ckpt
                    save_checkpoint(model, optimizer, scheduler, epoch, best_acc_c_ckpt_file_path, best_loss=val_loss, best_acc=best_acc)
                    logger.info(f">>>>> best acc {best_acc_c_ckpt_file_path}_{epoch}_{best_acc} saved......")    
            
            
            
            
            
            

        # if (epoch % args.test_freq == 0) | (epoch == args.epoches - 1):         
        #     model.eval()
        #     tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
        #     OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
        #     print("Epoch: {:03d} val_OA: {:.4f}val_Kappa: {:.4f}".format(epoch+1, OA2, Kappa2))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('ALL Training time {}'.format(total_time_str))




        
    # # prepare data
    # if args.dataset == 'Indian':
    #     data = loadmat('./data/IndianPine.mat')
    # elif args.dataset == 'Pavia':
    #     data = loadmat('./data/Pavia.mat')
    # elif args.dataset == 'Houston':
    #     data = loadmat('./data/Houston.mat')
    # else:
    #     raise ValueError("Unkknow dataset")
    # color_mat = loadmat('./data/AVIRIS_colormap.mat')
    # TR = data['TR']
    # TE = data['TE']
    # input = data['input'] #(145,145,200)
    # label = TR + TE
    # num_classes = np.max(TR)

    # color_mat_list = list(color_mat)
    # color_matrix = color_mat[color_mat_list[3]] #(17,3)
    # # normalize data by band norm
    # input_normalize = np.zeros(input.shape)
    # for i in range(input.shape[2]):
    #     input_max = np.max(input[:,:,i])
    #     input_min = np.min(input[:,:,i])
    #     input_normalize[:,:,i] = (input[:,:,i]-input_min)/(input_max-input_min)
    # # data size
    # height, width, band = input.shape
    # print("height={0},width={1},band={2}".format(height, width, band))
    # #-------------------------------------------------------------------------------
    # # obtain train and test data
    # total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = chooose_train_and_test_point(TR, TE, label, num_classes)
    # mirror_image = mirror_hsi(height, width, band, input_normalize, patch=args.patches)
    # x_train_band, x_test_band, x_true_band = train_and_test_data(mirror_image, band, total_pos_train, total_pos_test, total_pos_true, patch=args.patches, band_patch=args.band_patches)
    # y_train, y_test, y_true = train_and_test_label(number_train, number_test, number_true, num_classes)
    # #-------------------------------------------------------------------------------
    # # load data
    # x_train=torch.from_numpy(x_train_band.transpose(0,2,1)).type(torch.FloatTensor) #[695, 200, 7, 7]
    # y_train=torch.from_numpy(y_train).type(torch.LongTensor) #[695]
    # Label_train=Data.TensorDataset(x_train,y_train)
    # x_test=torch.from_numpy(x_test_band.transpose(0,2,1)).type(torch.FloatTensor) # [9671, 200, 7, 7]
    # y_test=torch.from_numpy(y_test).type(torch.LongTensor) # [9671]
    # Label_test=Data.TensorDataset(x_test,y_test)
    # x_true=torch.from_numpy(x_true_band.transpose(0,2,1)).type(torch.FloatTensor)
    # y_true=torch.from_numpy(y_true).type(torch.LongTensor)
    # Label_true=Data.TensorDataset(x_true,y_true)

    # label_train_loader=Data.DataLoader(Label_train,batch_size=args.batch_size,shuffle=True)
    # label_test_loader=Data.DataLoader(Label_test,batch_size=args.batch_size,shuffle=True)
    # label_true_loader=Data.DataLoader(Label_true,batch_size=100,shuffle=False)


    # if args.flag_test == 'test':
    #     if args.mode == 'ViT':
    #         model.load_state_dict(torch.load('./ViT.pt'))      
    #     elif (args.mode == 'CAF') & (args.patches == 1):
    #         model.load_state_dict(torch.load('./SpectralFormer_pixel.pt'))
    #     elif (args.mode == 'CAF') & (args.patches == 7):
    #         model.load_state_dict(torch.load('./SpectralFormer_patch.pt'))
    #     else:
    #         raise ValueError("Wrong Parameters") 
    #     model.eval()
    #     tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
    #     OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)

    #     # output classification maps
    #     pre_u = test_epoch(model, label_true_loader, criterion, optimizer)
    #     prediction_matrix = np.zeros((height, width), dtype=float)
    #     for i in range(total_pos_true.shape[0]):
    #         prediction_matrix[total_pos_true[i,0], total_pos_true[i,1]] = pre_u[i] + 1
    #     plt.subplot(1,1,1)
    #     plt.imshow(prediction_matrix, colors.ListedColormap(color_matrix))
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.show()
    #     savemat('matrix.mat',{'P':prediction_matrix, 'label':label})
    # elif args.flag_test == 'train':
    #     print("start training")
    #     tic = time.time()
    #     for epoch in range(args.epoches): 
    #         scheduler.step()

    #         # train model
    #         model.train()
    #         train_acc, train_obj, tar_t, pre_t = train_epoch(model, label_train_loader, criterion, optimizer)
    #         OA1, AA_mean1, Kappa1, AA1 = output_metric(tar_t, pre_t) 
    #         print("Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f}"
    #                         .format(epoch+1, train_obj, train_acc))
            

    #         if (epoch % args.test_freq == 0) | (epoch == args.epoches - 1):         
    #             model.eval()
    #             tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
    #             OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
    #             print("Epoch: {:03d} val_OA: {:.4f}val_Kappa: {:.4f}".format(epoch+1, OA2, Kappa2))

    #     toc = time.time()
    #     print("Running Time: {:.2f}".format(toc-tic))
    #     print("**************************************************")

    # print("Final result:")
    # print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2))
    # print(AA2)
    # print("**************************************************")
    # print("Parameter:")
    # print_args(vars(args))

# def print_args(args):
#     for k, v in zip(args.keys(), args.values()):
#         print("{0}: {1}".format(k,v))









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



