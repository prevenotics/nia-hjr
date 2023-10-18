import torch
import numpy as np
from utils.utils import make_output_directory, load_checkpoint_files, save_checkpoint, chooose_train_and_test_point, mirror_hsi, gain_neighborhood_pixel, gain_neighborhood_band, train_and_test_data, train_and_test_label, accuracy, output_metric, cal_results


class AvgrageMeter(object):
    
  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt

# train model
def train_epoch(model, train_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])    
    pixel_batch = 1024
    sampling_num = 2 
    for idx, batch in enumerate(train_loader):
        image = batch["image"].cuda(non_blocking=True)
        target = batch["label"].cuda(non_blocking=True)
        # reshaped_image = image.permute(0, 2, 3, 1).contiguous().view(-1, 200, 1)
        reshaped_image = image[:, :, ::sampling_num, ::sampling_num].permute(0, 2, 3, 1).contiguous().view(-1, 200, 1)
        reshaped_target = target.view(-1)
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
        
        for i in range(0, shuffled_image.size(0), pixel_batch):
            pixel_batch_image = shuffled_image[i:i+pixel_batch]
            pixel_batch_target = shuffled_target[i:i+pixel_batch]
            optimizer.zero_grad()
            batch_pred = model(pixel_batch_image)
            loss = criterion(batch_pred, pixel_batch_target)
            loss_val = loss.item()
            print(f"{loss_val}")
            loss.backward()
            optimizer.step()       

            prec1, t, p = accuracy(batch_pred, pixel_batch_target, topk=(1,))
            n = pixel_batch_image.shape[0]
            objs.update(loss.data, n)
            top1.update(prec1[0].data, n)
            tar = np.append(tar, t.data.cpu().numpy())
            pre = np.append(pre, p.data.cpu().numpy())
        
    return top1.avg, objs.avg, tar, pre
        
    
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
def valid_epoch(model, valid_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()   

        batch_pred = model(batch_data)
        
        loss = criterion(batch_pred, batch_target)

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
        
    return tar, pre

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
