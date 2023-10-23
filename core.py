import torch
import time
import datetime
import numpy as np
from timm.utils import AverageMeter
from utils.utils import make_output_directory, load_checkpoint_files, save_checkpoint, chooose_train_and_test_point, mirror_hsi, gain_neighborhood_pixel, gain_neighborhood_band, train_and_test_data, train_and_test_label, accuracy, output_metric, cal_results


# class AvgrageMeter(object):
    
#   def __init__(self):
#     self.reset()

#   def reset(self):
#     self.avg = 0
#     self.sum = 0
#     self.cnt = 0

#   def update(self, val, n=1):
#     self.sum += val * n
#     self.cnt += n
#     self.avg = self.sum / self.cnt

# train model
def train_epoch(model, train_loader, criterion, optimizer, lr_scheduler, epoch, num_epochs, logger, print_freq=1000):
    # objs = AverageMeter() #loss
    loss_meter = AverageMeter() #loss
    top1 = AverageMeter()
    pixel_batch_time=AverageMeter()
    batch_time = AverageMeter()
    lr_meter = AverageMeter()
    OA_meter = AverageMeter()
    Kappa_meter = AverageMeter()
    AA_meter = AverageMeter()
    tar = np.array([])
    pre = np.array([])    
    pixel_batch = 1024
    sampling_num = 2 
    
    num_steps = len(train_loader)
    
    
    batch_start = time.time()
    batch_end = time.time()
    pixel_batch_end = time.time()
    
    for idx, batch in enumerate(train_loader):
        image = batch["image"].cuda(non_blocking=True)
        target = batch["label"].cuda(non_blocking=True)
        # reshaped_image = image.permute(0, 2, 3, 1).contiguous().view(-1, 200, 1)
        # reshaped_image = image[:, :, ::sampling_num, ::sampling_num].permute(0, 2, 3, 1).contiguous().view(-1, 200, 1)
        
        # data = np.zeros((sample_point.shape[0], args.patch, args.patch, args.band), dtype=float)
        # for i in range(sample_point.shape[0]):
        #     data[i,:,:,:] = gain_neighborhood_pixel(image, sample_point, i, args.patch)
        # data = gain_neighborhood_band(data, args.band, args.band_patch, args.patch)
        reshaped_image = image.contiguous().view(-1,200,image.shape[3])
        reshaped_target = target.contiguous().view(-1)
        
        # reshaped_image = image[:,:,sample_point[:,0],sample_point[:,1]].permute(0,2,1).contiguous().view(-1, 200, 1)
        # reshaped_target = target[:,sample_point[:,0],sample_point[:,1]].view(-1)
        
        num_pixels = reshaped_image.size(0)
        random_order = torch.randperm(num_pixels)
        shuffled_image = reshaped_image[random_order]
        shuffled_target = reshaped_target[random_order]
        # optimizer.zero_grad()
        # batch_pred = model(shuffled_image)
        # loss = criterion(batch_pred, shuffled_target)
        # loss_val = loss.item()
        # print(f"{loss_val}")
        # loss.backward()
        # optimizer.step()      
        pixel_batch_start=time.time()
        for i in range(0, shuffled_image.size(0), pixel_batch):
            pixel_batch_image = shuffled_image[i:i+pixel_batch,:,:]
            pixel_batch_target = shuffled_target[i:i+pixel_batch]
            optimizer.zero_grad()
            batch_pred = model(pixel_batch_image)
            loss = criterion(batch_pred, pixel_batch_target)
            loss_val = loss.item()
            # print(f"{loss_val}")
            loss.backward()
            optimizer.step()       
            lr_scheduler.step()

            prec1, t, p = accuracy(batch_pred, pixel_batch_target, topk=(1,))
            n = pixel_batch_image.shape[0]
            loss_meter.update(loss.data, n)
            # loss_meter.update(loss.item(), n)
            lr_meter.update(optimizer.param_groups[0]["lr"])
            top1.update(prec1[0].data, n)
            tar = np.append(tar, t.data.cpu().numpy())
            pre = np.append(pre, p.data.cpu().numpy())
            pixel_batch_time.update(time.time() - pixel_batch_end)
            pixel_batch_end = time.time()
             
        # OA, AA_mean, Kappa, AA = output_metric(tar, pre)  #최종 OA, Kappa 값은 testset의 전체 픽셀레벨로 따로 구해야함. 이건 추이를 보기 위함
        OA, Kappa = output_metric(tar, pre)  #최종 OA, Kappa 값은 testset의 전체 픽셀레벨로 따로 구해야함. 이건 추이를 보기 위함
        OA_meter.update(OA,1)
        Kappa_meter.update(Kappa,1)
        # AA_meter.update(AA,1)
        
        batch_time.update(time.time() - batch_end)
        batch_end = time.time()
      
      
            
        if idx % print_freq == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
            f'Train: [{epoch}/{num_epochs}][{idx}/{num_steps}]\t'
            f'eta {datetime.timedelta(seconds=int(etas))}\t'
            f'PixelBatchTime {pixel_batch_time.val:.8f} ({pixel_batch_time.avg:.8f})\t'
            f'BatchTime {batch_time.val:.8f} ({batch_time.avg:.8f})\t'
            f'loss {loss_meter.val:.8f} ({loss_meter.avg:.8f})\t'  
            # f'cls_loss {cls_loss_meter.val:.8f} ({cls_loss_meter.avg:.8f})\t' 
            # f'dice_loss {dice_loss_meter.val:.8f} ({dice_loss_meter.avg:.8f})\t' 
            # f'dice_coef {dice_coef_meter.val:.8f} ({dice_coef_meter.avg:.8f})\t' 
            # f'train_acc_corr {acc_c_meter.val:.8f} ({acc_c_meter.avg:.8f})\t' 
            # f'train_acc_r {acc_r_meter.val:.8f} ({acc_r_meter.avg:.8f})\t' 
            f'grad_norm(lr) {lr_meter.val:.8f} ({lr_meter.avg:.8f})\t'
            f'mem {memory_used:.0f}MB')
    
    
    epoch_time = time.time() - batch_start
    
    return [top1, loss_meter, OA_meter, Kappa_meter, lr_meter]
        


def valid_epoch(model, train_loader, criterion, epoch, num_epochs, logger, print_freq=1000):
    # objs = AverageMeter() #loss
    loss_meter = AverageMeter() #loss
    top1 = AverageMeter()
    pixel_batch_time=AverageMeter()
    batch_time = AverageMeter()
    lr_meter = AverageMeter()
    OA_meter = AverageMeter()
    Kappa_meter = AverageMeter()
    AA_meter = AverageMeter()
    tar = np.array([])
    pre = np.array([])    
    pixel_batch = 1024
    sampling_num = 2 
    
    num_steps = len(train_loader)
    
    
    batch_start = time.time()
    batch_end = time.time()
    pixel_batch_end = time.time()
    
    for idx, batch in enumerate(train_loader):
        image = batch["image"].cuda(non_blocking=True)
        target = batch["label"].cuda(non_blocking=True)
        
        reshaped_image = image.contiguous().view(-1,200,image.shape[3])
        reshaped_target = target.contiguous().view(-1)
        
        
        num_pixels = reshaped_image.size(0)
        random_order = torch.randperm(num_pixels)
        shuffled_image = reshaped_image[random_order]
        shuffled_target = reshaped_target[random_order]

        pixel_batch_start=time.time()
        for i in range(0, shuffled_image.size(0), pixel_batch):
            pixel_batch_image = shuffled_image[i:i+pixel_batch,:,:]
            pixel_batch_target = shuffled_target[i:i+pixel_batch]

            batch_pred = model(pixel_batch_image)
            loss = criterion(batch_pred, pixel_batch_target)

            prec1, t, p = accuracy(batch_pred, pixel_batch_target, topk=(1,))
            n = pixel_batch_image.shape[0]
            loss_meter.update(loss.data, n)
            # loss_meter.update(loss.item(), n)
            # lr_meter.update(optimizer.param_groups[0]["lr"])
            top1.update(prec1[0].data, n)
            tar = np.append(tar, t.data.cpu().numpy())
            pre = np.append(pre, p.data.cpu().numpy())
            pixel_batch_time.update(time.time() - pixel_batch_end)
            pixel_batch_end = time.time()
             
        # OA, AA_mean, Kappa, AA = output_metric(tar, pre)  #최종 OA, Kappa 값은 testset의 전체 픽셀레벨로 따로 구해야함. 이건 추이를 보기 위함
        OA, Kappa = output_metric(tar, pre)  #최종 OA, Kappa 값은 testset의 전체 픽셀레벨로 따로 구해야함. 이건 추이를 보기 위함
        OA_meter.update(OA,1)
        Kappa_meter.update(Kappa,1)
        # AA_meter.update(AA,1)
        
        batch_time.update(time.time() - batch_end)
        batch_end = time.time()
      
      
            
        if idx % print_freq == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
            f'Valid: [{epoch}/{num_epochs}][{idx}/{num_steps}]\t'
            f'eta {datetime.timedelta(seconds=int(etas))}\t'
            f'PixelBatchTime {pixel_batch_time.val:.8f} ({pixel_batch_time.avg:.8f})\t'
            f'BatchTime {batch_time.val:.8f} ({batch_time.avg:.8f})\t'
            f'loss {loss_meter.val:.8f} ({loss_meter.avg:.8f})\t'  
            # f'cls_loss {cls_loss_meter.val:.8f} ({cls_loss_meter.avg:.8f})\t' 
            # f'dice_loss {dice_loss_meter.val:.8f} ({dice_loss_meter.avg:.8f})\t' 
            # f'dice_coef {dice_coef_meter.val:.8f} ({dice_coef_meter.avg:.8f})\t' 
            # f'train_acc_corr {acc_c_meter.val:.8f} ({acc_c_meter.avg:.8f})\t' 
            # f'train_acc_r {acc_r_meter.val:.8f} ({acc_r_meter.avg:.8f})\t' 
            # f'grad_norm(lr) {lr_meter.val:.8f} ({lr_meter.avg:.8f})\t'
            f'mem {memory_used:.0f}MB')
    
    
    epoch_time = time.time() - batch_start
    
    return [top1, loss_meter, OA_meter, Kappa_meter]

#-------------------------------------------------------------------------------
# def train_epoch(model, train_loader, criterion, optimizer):
#     objs = AvgrageMeter()
#     top1 = AvgrageMeter()
#     tar = np.array([])
#     pre = np.array([])
#     for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
#         batch_data = batch_data.cuda()
#         batch_target = batch_target.cuda()   

#         optimizer.zero_grad()
#         batch_pred = model(batch_data)
#         loss = criterion(batch_pred, batch_target)
#         loss.backward()
#         optimizer.step()       

#         prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
#         n = batch_data.shape[0]
#         objs.update(loss.data, n)
#         top1.update(prec1[0].data, n)
#         tar = np.append(tar, t.data.cpu().numpy())
#         pre = np.append(pre, p.data.cpu().numpy())
#     return top1.avg, objs.avg, tar, pre


# validate model
# def valid_epoch(model, valid_loader, criterion, optimizer):
#     objs = AvgrageMeter()
#     top1 = AvgrageMeter()
#     tar = np.array([])
#     pre = np.array([])
#     for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
#         batch_data = batch_data.cuda()
#         batch_target = batch_target.cuda()   

#         batch_pred = model(batch_data)
        
#         loss = criterion(batch_pred, batch_target)

#         prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
#         n = batch_data.shape[0]
#         objs.update(loss.data, n)
#         top1.update(prec1[0].data, n)
#         tar = np.append(tar, t.data.cpu().numpy())
#         pre = np.append(pre, p.data.cpu().numpy())
        
#     return tar, pre

def test_epoch(model, test_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(test_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()   

        batch_pred = model(batch_data)

        _, pred = batch_pred.topk(1, 1, True, True)
        pp = pred.squeeze()
        pre = np.append(pre, pp.data.cpu().numpy())
    return pre
