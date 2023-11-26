import torch
from torchvision import transforms
import time
import datetime
import numpy as np
import math
from timm.utils import AverageMeter
import utils.utils as util
from PIL import Image, ImageDraw, ImageFont
import os
import cv2
from collections import Counter


def train_epoch(model, tar, pre, train_loader, criterion, optimizer, lr_scheduler, epoch, cfg, logger, print_freq=1000):
    # objs = AverageMeter() #loss
    loss_meter = AverageMeter() #loss
    top1 = AverageMeter()
    pixel_batch_time=AverageMeter()
    batch_time = AverageMeter()
    lr_meter = AverageMeter()
    OA_meter = AverageMeter()
    Kappa_meter = AverageMeter()
    pixel_batch = cfg['train_param']['pixel_batch']
    num_epochs = cfg['train_param']['epoch']
    band =cfg['image_param']['band']
    num_steps = len(train_loader)
    
    
    batch_start = time.time()
    batch_end = time.time()
    pixel_batch_end = time.time()
    
    for idx, batch in enumerate(train_loader):
        image = batch["image"].cuda(non_blocking=True)
        target = batch["label"].cuda(non_blocking=True)
        reshaped_image = image.contiguous().view(-1,band,image.shape[3])
        reshaped_target = target.contiguous().view(-1)
        num_pixels = reshaped_image.size(0)
        random_order = torch.randperm(num_pixels)
        shuffled_image = reshaped_image[random_order]
        shuffled_target = reshaped_target[random_order]
        
        pixel_batch_start=time.time()
        for i in range(0, shuffled_image.size(0), pixel_batch):
            pixel_batch_image = shuffled_image[i:i+pixel_batch,:,:]
            pixel_batch_target = shuffled_target[i:i+pixel_batch]
            optimizer.zero_grad()
            batch_pred = model(pixel_batch_image)
            loss = criterion(batch_pred, pixel_batch_target)
            loss_val = loss.item()
            loss.backward()
            optimizer.step()       
            lr_scheduler.step()

            prec1, t, p, ignore_cnt = util.accuracy(batch_pred, pixel_batch_target, topk=(1,))                
            n = pixel_batch_image.shape[0] - ignore_cnt
            loss_meter.update(loss.data, n)
            lr_meter.update(optimizer.param_groups[0]["lr"])
            top1.update(prec1[0].data, n)
            tar = np.append(tar, t.data.cpu().numpy())
            pre = np.append(pre, p.data.cpu().numpy())
            pixel_batch_time.update(time.time() - pixel_batch_end)
            pixel_batch_end = time.time()
        
        
        batch_time.update(time.time() - batch_end)
        batch_end = time.time()
      
        OA, Kappa = util.output_metric(tar, pre)     
            
        if idx % print_freq == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
            f'Train: [{epoch}/{num_epochs}][{idx}/{num_steps}]\t'                        
            f'eta {datetime.timedelta(seconds=int(etas))}\t'            
            f'OA, kappa: {OA:.4f}, {Kappa:.4f}\t'
            f'loss {loss_meter.val:.2f} ({loss_meter.avg:.2f})\t'              
            f'grad_norm(lr) {lr_meter.val:.2f} ({lr_meter.avg:.2f})\t')
            
    
    OA, Kappa = util.output_metric(tar, pre)  
    OA_meter.update(OA,1)
    Kappa_meter.update(Kappa,1)
    
    epoch_time = time.time() - batch_start
    
    return [top1, loss_meter, OA_meter, Kappa_meter, lr_meter]


def valid_epoch(model, train_loader, criterion, epoch, cfg, logger, print_freq=1000):
    loss_meter = AverageMeter() 
    top1 = AverageMeter()
    pixel_batch_time=AverageMeter()
    batch_time = AverageMeter()
    lr_meter = AverageMeter()
    OA_meter = AverageMeter()
    Kappa_meter = AverageMeter()
    AA_meter = AverageMeter()
    tar = np.array([])
    pre = np.array([])        
    pixel_batch = cfg['train_param']['pixel_batch']
    sampling_num = 2 
    num_epochs = cfg['train_param']['epoch']
    band =cfg['image_param']['band']
    num_steps = len(train_loader)
    
    
    batch_start = time.time()
    batch_end = time.time()
    pixel_batch_end = time.time()
    
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(train_loader):
            image = batch["image"].cuda(non_blocking=True)
            target = batch["label"].cuda(non_blocking=True)
            
            reshaped_image = image.contiguous().view(-1,band,image.shape[3])
            reshaped_target = target.contiguous().view(-1)
            
            
            num_pixels = reshaped_image.size(0)            
            shuffled_image = reshaped_image
            shuffled_target = reshaped_target

            pixel_batch_start=time.time()
            for i in range(0, shuffled_image.size(0), pixel_batch):
                pixel_batch_image = shuffled_image[i:i+pixel_batch,:,:]
                pixel_batch_target = shuffled_target[i:i+pixel_batch]

                batch_pred = model(pixel_batch_image)
                loss = criterion(batch_pred, pixel_batch_target)

                prec1, t, p, ignore_cnt = util.accuracy(batch_pred, pixel_batch_target, topk=(1,))                
                n = pixel_batch_image.shape[0] - ignore_cnt
                loss_meter.update(loss.data, n)                
                top1.update(prec1[0].data, n)
                tar = np.append(tar, t.data.cpu().numpy())
                pre = np.append(pre, p.data.cpu().numpy())
                
                pixel_batch_time.update(time.time() - pixel_batch_end)
                pixel_batch_end = time.time()
            
            OA, Kappa = util.output_metric(tar, pre) 
            OA_meter.update(OA,1)
            Kappa_meter.update(Kappa,1)
            
            batch_time.update(time.time() - batch_end)
            batch_end = time.time()
        
        
                
            if idx % print_freq == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                etas = batch_time.avg * (num_steps - idx)
                logger.info(
                f'Valid: [{epoch}/{num_epochs}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))}\t'                
                f'OA, kappa: {OA:.4f}, {Kappa:.4f}\t'
                f'loss {loss_meter.val:.8f} ({loss_meter.avg:.8f})\t'                  
                f'mem {memory_used:.0f}MB')
    
    
    epoch_time = time.time() - batch_start
    
    return [top1, loss_meter, OA_meter, Kappa_meter]


def test_epoch(model, test_loader, cfg, logger):
    top1 = AverageMeter()
    pixel_batch_time=AverageMeter()
    batch_time = AverageMeter()
    lr_meter = AverageMeter()
    OA_meter = AverageMeter()
    Kappa_meter = AverageMeter()
    AA_meter = AverageMeter()
    pixel_batch = cfg['train_param']['pixel_batch']
    sampling_num = 2 
    num_epochs = cfg['train_param']['epoch']
    band =cfg['image_param']['band']
    npy_path=cfg['test_param']['out_npy_path']
    num_steps = len(test_loader)
    cls_name = util.class_name()
    ignore_cls = 30
    
    batch_start = time.time()
    batch_end = time.time()
    pixel_batch_end = time.time()
    
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    formatted_time_for_filename = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    
    testlogtxt=os.path.join(cfg['test_param']['out_res_path'], f'tes_log_{formatted_time_for_filename}.txt')
    with open(testlogtxt, "w") as log:
        log.write(f"Start Time [{formatted_time}]\n")
    
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            image = batch["image"].cuda(non_blocking=True)            
            origin_image = batch["origin_image"].cuda(non_blocking=True)
            target = batch["label"].cuda(non_blocking=True)
            path = batch["path"]
            
            # B 470nm = 15 // G 540nm = 39 // R 660nm = 80            
            image_size = int(image.shape[1] **(1/2))
            batch_size = image.shape[0]     
            reshaped_image = image.contiguous().view(-1,band,image.shape[3])
            reshaped_target = target.contiguous().view(-1)
                         
            num_pixels = reshaped_image.size(0)
            tar_one = np.array([])
            pre_one = np.array([])                

            pixel_batch_start=time.time()
            for i in range(0, reshaped_image.size(0), pixel_batch):
                pixel_batch_image = reshaped_image[i:i+pixel_batch,:,:]
                pixel_batch_target = reshaped_target[i:i+pixel_batch]

                batch_pred = model(pixel_batch_image)
                prec1, t, p, ignore_cnt = util.accuracy(batch_pred, pixel_batch_target, topk=(1,))                
                n = pixel_batch_image.shape[0] - ignore_cnt
                top1.update(prec1[0].data, n)
                tar_one = np.append(tar_one, t.data.cpu().numpy())
                pre_one = np.append(pre_one, p.data.cpu().numpy())
                pixel_batch_time.update(time.time() - pixel_batch_end)
                pixel_batch_end = time.time()

            
            label_pre =pre_one.reshape(batch_size,image_size,image_size)
            label_tar= tar_one.reshape(batch_size,image_size,image_size)
            
            
            if not cfg['test_param']['save_img']:
                OA_list = []
                Kappa_list = []
                for i in range(batch_size):
                    label_tar_i = (label_tar[i, :, :]).astype(np.uint8)
                    label_pre_i = (label_pre[i, :, :]).astype(np.uint8)
                    
                    label_pre_i = cv2.medianBlur(label_pre_i, 9)                    
                    OA_one, Kappa_one = util.output_metric(label_tar_i.reshape(-1), label_pre_i.reshape(-1))                
                    
                    OA_list.append(OA_one)
                    Kappa_list.append(Kappa_one)
                    # index = idx*batch_size+i
                    img_name = os.path.basename(path[i]).split('.')[0]
                    npy_path_final = os.path.join(npy_path, f'{formatted_time_for_filename}_{img_name}.npy')
                    np.save(npy_path, np.vstack((label_tar_i.reshape(-1), label_pre_i.reshape(-1))))
                
                OA_batch = np.array(OA_list)
                Kappa_batch = np.array(Kappa_list)            
            else:                       
                image_margin = 5
                OA_list = []
                Kappa_list = []              
                for i in range(batch_size):
                    basename=os.path.basename(path[i])
                    if basename[2] =='L' or basename[2] =='U':
                        res_image = origin_image[i, :, :].reshape(image_size, image_size, band)[:, :, [80, 39, 15]].cpu().numpy()
                    elif basename[2] =='D':
                        res_image = origin_image[i, :, :].reshape(image_size, image_size, band)[:, :, [58, 28, 11]].cpu().numpy()
                        
                    res_image = (res_image * 255).astype(np.uint8)
                    
                    label_tar_i = (label_tar[i, :, :]).astype(np.uint8)
                    label_pre_i = (label_pre[i, :, :]).astype(np.uint8)
                    
                    label_pre_i = cv2.medianBlur(label_pre_i, 9)          
                    OA_one, Kappa_one = util.output_metric(label_tar_i.reshape(-1), label_pre_i.reshape(-1))                          
                    OA_list.append(OA_one)
                    Kappa_list.append(Kappa_one)
                    img_name = os.path.basename(path[i]).split('.')[0]
                    npy_path_final = os.path.join(npy_path, f'{formatted_time_for_filename}_{img_name}.npy')
                    np.save(npy_path_final, np.vstack((label_tar_i.reshape(-1), label_pre_i.reshape(-1))))
                    
                    tar_unique, tar_cnt = np.unique(label_tar_i[label_tar_i !=30], return_counts=True)
                    sorted_indices = np.argsort(-tar_cnt)
                    tar_unique = tar_unique[sorted_indices]
                    tar_cnt = tar_cnt[sorted_indices]

                    pre_unique, pre_cnt = np.unique(label_pre_i, return_counts=True)
                    sorted_indices = np.argsort(-pre_cnt)
                    pre_unique = pre_unique[sorted_indices]
                    pre_cnt = pre_cnt[sorted_indices]
                                       
                    label_tar_i = np.repeat(label_tar_i[:, :, np.newaxis] * 8, 3, axis=2)
                    label_pre_i = np.repeat(label_pre_i[:, :, np.newaxis] * 8, 3, axis=2)       
                    
                    ignore_idxs = np.all(label_tar_i == [240, 240, 240], axis=2)  # 값이 30 * 8 = 240인 부분 찾기
                    label_tar_i[ignore_idxs] = [255, 240, 0]
                    label_pre_i[ignore_idxs] = [255, 240, 0]
                    
                    result_img = np.zeros((image_size + 100, image_size * 3 + image_margin * 2, 3), dtype=np.uint8)
                    
                    
                    result_img = Image.new('RGB', (image_size*3+image_margin*2, image_size+100))                
                    result_img.paste(Image.fromarray(res_image), (0,0))
                    result_img.paste(Image.fromarray(label_tar_i), (image_size+image_margin,0))
                    result_img.paste(Image.fromarray(label_pre_i), (image_size*2+image_margin,0))
                    result_img_draw = ImageDraw.Draw(result_img)
                    font = ImageFont.truetype("ARIAL.TTF", 20)
                    result_img_draw.text((10,image_size+5), f" OA   : {OA_one:.3f}", (0,255,0), font=font)
                    result_img_draw.text((10,image_size+35), f"Kappa: {Kappa_one:.3f}", (0,255,0), font=font)
                    font = ImageFont.truetype("MALGUN.TTF", 15)
                    
                    max_length = max(len(tar_unique), len(pre_unique))
                    for ii in range(max_length):
                        if ii <len(tar_unique):
                            text_x = image_size +15
                            text_y = image_size + 3 + 20*(ii)
                            text = f'GT[{ii}] : {cls_name[tar_unique[ii]]}'
                            result_img_draw.text((text_x,text_y), text, (0,255,0), font=font)
                        if ii < len(pre_unique):
                            text_x = image_size *2+15
                            text_y = image_size + 3 + 20*(ii)
                            text = f'Result[{ii}] : {cls_name[pre_unique[ii]]}'
                            result_img_draw.text((text_x,text_y), text, (0,255,0), font=font)
                    
                    img_name = os.path.basename(path[i]).split('.')[0]
                    result_img_path = os.path.join(cfg['test_param']['out_img_path'], f'{formatted_time_for_filename}_{img_name}.jpg')
                    result_img.save(result_img_path)
                
                OA_batch = np.array(OA_list)
                Kappa_batch = np.array(Kappa_list)  
                
            batch_time.update(time.time() - batch_end)
            batch_end = time.time()

            for i in range(batch_size):     
                with open(testlogtxt, "a") as file:                    
                    path_one = path[i]
                    if OA_batch[i] > 0.8 and Kappa_batch[i] >0.7:
                        file.write(f'{path_one}\tOA\t{OA_batch[i]:0.3f}\tkappa\t{Kappa_batch[i]:0.3f}\tO\tO\n')
                    elif OA_batch[i] > 0.8 and Kappa_batch[i] <= 0.7: 
                        file.write(f'{path_one}\tOA\t{OA_batch[i]:0.3f}\tkappa\t{Kappa_batch[i]:0.3f}\tO\t \n')
                    elif OA_batch[i] <= 0.8 and Kappa_batch[i] > 0.7: 
                        file.write(f'{path_one}\tOA\t{OA_batch[i]:0.3f}\tkappa\t{Kappa_batch[i]:0.3f}\t \tO\n')
                    else:
                        file.write(f'{path_one}\tOA\t{OA_batch[i]:0.3f}\tkappa\t{Kappa_batch[i]:0.3f}\t \t \n')
                
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                etas = batch_time.avg * (num_steps - idx)
                logger.info(
                f'Test: [{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))}\t'
                f'OA, kappa: {OA_batch[i]:.4f}, {Kappa_batch[i]:.4f}')

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")    
    print(f"End Time [{formatted_time}]\n")
            
    with open(testlogtxt, "a") as file:
        file.write(f"End Time [{formatted_time}]\n")    
    
    epoch_time = time.time() - batch_start
    
    result = []
    cnt = 0
    for root, dirs, files in os.walk(npy_path):
        # if "02.수중 및 지상 초분광" in root:
            for image_filename in files:
                if image_filename.endswith('.npy'):
                    temp = np.load(os.path.join(npy_path, image_filename))
                    result.append(temp)
                    

    temp = np.array(result)
    temp = temp.transpose(1, 2, 0).reshape(2, -1)
    OA, kappa = util.output_metric_with_savefig(temp[0,:], temp[1,:], cfg['test_param']['out_res_path'])
    print(f'========Total : OA:{OA:.5f} Kappa:{kappa:.5f}========')
    
    # return [top1, OA_meter, Kappa_meter]
    return top1



def test_epoch_online(model, test_loader, band, imgtype, cfg, logger):
    top1 = AverageMeter()
    pixel_batch_time=AverageMeter()
    batch_time = AverageMeter()
    lr_meter = AverageMeter()
    OA_meter = AverageMeter()
    Kappa_meter = AverageMeter()
    AA_meter = AverageMeter()
    pixel_batch = cfg['train_param']['pixel_batch']
    sampling_num = 2 
    num_epochs = cfg['train_param']['epoch']
    # npy_path=cfg['test_param']['out_npy_path']
    num_steps = len(test_loader)
    cls_name = util.class_name()
    ignore_cls = 30
    
    batch_start = time.time()
    batch_end = time.time()
    pixel_batch_end = time.time()
    
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    formatted_time_for_filename = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    
    testlogtxt=os.path.join(cfg['test_param']['out_res_path'], f'tes_log_{formatted_time_for_filename}.txt')
    with open(testlogtxt, "w") as log:
        log.write(f"Start Time [{formatted_time}]\n")
    
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            image = batch["image"].cuda(non_blocking=True)            
            origin_image = batch["origin_image"].cuda(non_blocking=True)
            target = batch["label"].cuda(non_blocking=True)
            path = batch["path"]
            
            # B 470nm = 15 // G 540nm = 39 // R 660nm = 80            
            image_size = int(image.shape[1] **(1/2))
            batch_size = image.shape[0]     
            reshaped_image = image.contiguous().view(-1,band,image.shape[3])
            reshaped_target = target.contiguous().view(-1)
                         
            num_pixels = reshaped_image.size(0)
            tar_one = np.array([])
            pre_one = np.array([])                

            pixel_batch_start=time.time()
            for i in range(0, reshaped_image.size(0), pixel_batch):
                pixel_batch_image = reshaped_image[i:i+pixel_batch,:,:]
                pixel_batch_target = reshaped_target[i:i+pixel_batch]

                batch_pred = model(pixel_batch_image)
                prec1, t, p, ignore_cnt = util.accuracy(batch_pred, pixel_batch_target, topk=(1,))                
                n = pixel_batch_image.shape[0] - ignore_cnt
                top1.update(prec1[0].data, n)
                tar_one = np.append(tar_one, t.data.cpu().numpy())
                pre_one = np.append(pre_one, p.data.cpu().numpy())
                pixel_batch_time.update(time.time() - pixel_batch_end)
                pixel_batch_end = time.time()

            
            label_pre =pre_one.reshape(batch_size,image_size,image_size)
            label_tar= tar_one.reshape(batch_size,image_size,image_size)
            
            if not cfg['test_param']['save_img']:
                OA_list = []
                Kappa_list = []
                for i in range(batch_size):
                    label_tar_i = (label_tar[i, :, :]).astype(np.uint8)
                    label_pre_i = (label_pre[i, :, :]).astype(np.uint8)
                    
                    label_pre_i = cv2.medianBlur(label_pre_i, 9)                    
                    OA_one, Kappa_one = util.output_metric(label_tar_i.reshape(-1), label_pre_i.reshape(-1))                
                    
                    OA_list.append(OA_one)
                    Kappa_list.append(Kappa_one)
                    # index = idx*batch_size+i
                    # img_name = os.path.basename(path[i]).split('.')[0]
                    # npy_path_final = os.path.join(npy_path, f'{formatted_time_for_filename}_{img_name}.npy')
                    # np.save(npy_path, np.vstack((label_tar_i.reshape(-1), label_pre_i.reshape(-1))))
                
                OA_batch = np.array(OA_list)
                Kappa_batch = np.array(Kappa_list)    
            
            else:
                image_margin = 5
                OA_list = []
                Kappa_list = []              
                for i in range(batch_size):         
                    basename=os.path.basename(path[i])     
                    if basename[2] =='L' or basename[2] =='U':
                        res_image = origin_image[i, :, :].reshape(image_size, image_size, band)[:, :, [80, 39, 15]].cpu().numpy()
                    elif basename[2] =='D':
                        res_image = origin_image[i, :, :].reshape(image_size, image_size, band)[:, :, [58, 28, 11]].cpu().numpy()
                    
                    res_image = (res_image * 255).astype(np.uint8)
                    
                    label_tar_i = (label_tar[i, :, :]).astype(np.uint8)
                    label_pre_i = (label_pre[i, :, :]).astype(np.uint8)
                    
                    label_pre_i = cv2.medianBlur(label_pre_i, 9)          
                    OA_one, Kappa_one = util.output_metric(label_tar_i.reshape(-1), label_pre_i.reshape(-1))                          
                    OA_list.append(OA_one)
                    Kappa_list.append(Kappa_one)
                    
                    
                    tar_unique, tar_cnt = np.unique(label_tar_i[label_tar_i !=30], return_counts=True)
                    sorted_indices = np.argsort(-tar_cnt)
                    tar_unique = tar_unique[sorted_indices]
                    tar_cnt = tar_cnt[sorted_indices]

                    pre_unique, pre_cnt = np.unique(label_pre_i, return_counts=True)
                    sorted_indices = np.argsort(-pre_cnt)
                    pre_unique = pre_unique[sorted_indices]
                    pre_cnt = pre_cnt[sorted_indices]
                                       
                    label_tar_i = np.repeat(label_tar_i[:, :, np.newaxis] * 8, 3, axis=2)
                    label_pre_i = np.repeat(label_pre_i[:, :, np.newaxis] * 8, 3, axis=2)       
                    
                    ignore_idxs = np.all(label_tar_i == [240, 240, 240], axis=2)  # 값이 30 * 8 = 240인 부분 찾기
                    label_tar_i[ignore_idxs] = [255, 240, 0]
                    label_pre_i[ignore_idxs] = [255, 240, 0]
                    
                    result_img = np.zeros((image_size + 100, image_size * 3 + image_margin * 2, 3), dtype=np.uint8)
                    
                    
                    result_img = Image.new('RGB', (image_size*3+image_margin*2, image_size+100))                
                    result_img.paste(Image.fromarray(res_image), (0,0))
                    result_img.paste(Image.fromarray(label_tar_i), (image_size+image_margin,0))
                    result_img.paste(Image.fromarray(label_pre_i), (image_size*2+image_margin,0))
                    result_img_draw = ImageDraw.Draw(result_img)
                    font = ImageFont.truetype("ARIAL.TTF", 20)
                    result_img_draw.text((10,image_size+5), f" OA   : {OA_one:.3f}", (0,255,0), font=font)
                    result_img_draw.text((10,image_size+35), f"Kappa: {Kappa_one:.3f}", (0,255,0), font=font)
                    font = ImageFont.truetype("MALGUN.TTF", 15)
                    
                    max_length = max(len(tar_unique), len(pre_unique))
                    for ii in range(max_length):
                        if ii <len(tar_unique):
                            text_x = image_size +15
                            text_y = image_size + 3 + 20*(ii)
                            text = f'GT[{ii}] : {cls_name[tar_unique[ii]]}'
                            result_img_draw.text((text_x,text_y), text, (0,255,0), font=font)
                        if ii < len(pre_unique):
                            text_x = image_size *2+15
                            text_y = image_size + 3 + 20*(ii)
                            text = f'Result[{ii}] : {cls_name[pre_unique[ii]]}'
                            result_img_draw.text((text_x,text_y), text, (0,255,0), font=font)
                    
                    img_name = os.path.basename(path[i]).split('.')[0]
                    result_img_path = os.path.join(cfg['test_param']['out_img_path'], f'{img_name}_{formatted_time_for_filename}.jpg')
                    result_img.save(result_img_path)
                
                OA_batch = np.array(OA_list)
                Kappa_batch = np.array(Kappa_list)  
                
            batch_time.update(time.time() - batch_end)
            batch_end = time.time()

            for i in range(batch_size):     
                with open(testlogtxt, "a") as file:                    
                    path_one = path[i]
                    if OA_batch[i] > 0.8 and Kappa_batch[i] >0.7:
                        file.write(f'{path_one}\tOA\t{OA_batch[i]:0.3f}\tkappa\t{Kappa_batch[i]:0.3f}\tO\tO\n')
                    elif OA_batch[i] > 0.8 and Kappa_batch[i] <= 0.7: 
                        file.write(f'{path_one}\tOA\t{OA_batch[i]:0.3f}\tkappa\t{Kappa_batch[i]:0.3f}\tO\t \n')
                    elif OA_batch[i] <= 0.8 and Kappa_batch[i] > 0.7: 
                        file.write(f'{path_one}\tOA\t{OA_batch[i]:0.3f}\tkappa\t{Kappa_batch[i]:0.3f}\t \tO\n')
                    else:
                        file.write(f'{path_one}\tOA\t{OA_batch[i]:0.3f}\tkappa\t{Kappa_batch[i]:0.3f}\t \t \n')
                
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                etas = batch_time.avg * (num_steps - idx)
                logger.info(
                f'Test: [{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))}\t'
                f'OA, kappa: {OA_batch[i]:.4f}, {Kappa_batch[i]:.4f}')

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")    
    print(f"End Time [{formatted_time}]\n")
            
    with open(testlogtxt, "a") as file:
        file.write(f"End Time [{formatted_time}]\n")    
    
    epoch_time = time.time() - batch_start
    
    # return [top1, OA_meter, Kappa_meter]
    return top1
