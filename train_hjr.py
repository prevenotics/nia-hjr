import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from scipy.io import loadmat
from scipy.io import savemat
from torch import optim
from torch.autograd import Variable
from vit_pytorch import ViT
from utils.utils import make_output_directory, load_checkpoint_files, save_checkpoint, chooose_train_and_test_point, mirror_hsi, gain_neighborhood_pixel, gain_neighborhood_band, train_and_test_data, train_and_test_label, accuracy, output_metric, cal_results
from utils.logger import create_logger
from tensorboardX import SummaryWriter
from data.hjr_dataset import HJRDataset

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import time
import os

os.chdir("/root/work/hjr/IEEE_TGRS_SpectralFormer")

pixel_batch = 512

parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['Indian', 'Pavia', 'Houston'], default='Indian', help='dataset to use')
parser.add_argument('--flag_test', choices=['test', 'train'], default='train', help='testing mark')
parser.add_argument('--mode', choices=['ViT', 'CAF'], default='ViT', help='mode choice')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
# parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--batch_size', type=int, default=2, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=5, help='number of evaluation')
parser.add_argument('--patches', type=int, default=1, help='number of patches')
parser.add_argument('--band_patches', type=int, default=1, help='number of related band')
parser.add_argument('--epoches', type=int, default=300, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

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
    
    for idx, batch in enumerate(train_loader):
        image = batch["image"].cuda(non_blocking=True)
        target = batch["label"].cuda(non_blocking=True)
        reshaped_image = image.permute(0, 2, 3, 1).contiguous().view(-1, 200, 1)
        reshaped_target = target.view(-1)
        num_pixels = reshaped_image.size(0)
        random_order = torch.randperm(num_pixels)
        shuffled_image = reshaped_image[random_order]
        shuffled_target = reshaped_target[random_order]
        
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

# Parameter Setting
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False



imgtype ='RA'
csv_path= '/root/work/hjr/dataset/train.csv'
dataset = HJRDataset(csv_file=csv_path, imgtype=imgtype)
data_loader=Data.DataLoader(dataset,batch_size=args.batch_size,shuffle=True)

band =200
num_classes = 31
#-------------------------------------------------------------------------------
# create model
model = ViT(
    image_size = args.patches,
    near_band = args.band_patches,
    num_patches = band,
    num_classes = num_classes,
    dim = 64,
    depth = 5,
    heads = 4,
    mlp_dim = 8,
    dropout = 0.1,
    emb_dropout = 0.1,
    mode = args.mode
)
model = model.cuda()
# criterion
criterion = nn.CrossEntropyLoss(ignore_index=30).cuda()
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches//10, gamma=args.gamma)
#-------------------------------------------------------------------------------

for epoch in range(args.epoches): 
    scheduler.step()

    # train model
    model.train()
    train_acc, train_obj, tar_t, pre_t = train_epoch(model, data_loader, criterion, optimizer)
    OA1, AA_mean1, Kappa1, AA1 = output_metric(tar_t, pre_t) 
    print("Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f}"
                    .format(epoch+1, train_obj, train_acc))
    

    # if (epoch % args.test_freq == 0) | (epoch == args.epoches - 1):         
    #     model.eval()
    #     tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
    #     OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
    #     print("Epoch: {:03d} val_OA: {:.4f}val_Kappa: {:.4f}".format(epoch+1, OA2, Kappa2))

toc = time.time()
print("Running Time: {:.2f}".format(toc-tic))
print("**************************************************")










    
# prepare data
if args.dataset == 'Indian':
    data = loadmat('./data/IndianPine.mat')
elif args.dataset == 'Pavia':
    data = loadmat('./data/Pavia.mat')
elif args.dataset == 'Houston':
    data = loadmat('./data/Houston.mat')
else:
    raise ValueError("Unkknow dataset")
color_mat = loadmat('./data/AVIRIS_colormap.mat')
TR = data['TR']
TE = data['TE']
input = data['input'] #(145,145,200)
label = TR + TE
num_classes = np.max(TR)

color_mat_list = list(color_mat)
color_matrix = color_mat[color_mat_list[3]] #(17,3)
# normalize data by band norm
input_normalize = np.zeros(input.shape)
for i in range(input.shape[2]):
    input_max = np.max(input[:,:,i])
    input_min = np.min(input[:,:,i])
    input_normalize[:,:,i] = (input[:,:,i]-input_min)/(input_max-input_min)
# data size
height, width, band = input.shape
print("height={0},width={1},band={2}".format(height, width, band))
#-------------------------------------------------------------------------------
# obtain train and test data
total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = chooose_train_and_test_point(TR, TE, label, num_classes)
mirror_image = mirror_hsi(height, width, band, input_normalize, patch=args.patches)
x_train_band, x_test_band, x_true_band = train_and_test_data(mirror_image, band, total_pos_train, total_pos_test, total_pos_true, patch=args.patches, band_patch=args.band_patches)
y_train, y_test, y_true = train_and_test_label(number_train, number_test, number_true, num_classes)
#-------------------------------------------------------------------------------
# load data
x_train=torch.from_numpy(x_train_band.transpose(0,2,1)).type(torch.FloatTensor) #[695, 200, 7, 7]
y_train=torch.from_numpy(y_train).type(torch.LongTensor) #[695]
Label_train=Data.TensorDataset(x_train,y_train)
x_test=torch.from_numpy(x_test_band.transpose(0,2,1)).type(torch.FloatTensor) # [9671, 200, 7, 7]
y_test=torch.from_numpy(y_test).type(torch.LongTensor) # [9671]
Label_test=Data.TensorDataset(x_test,y_test)
x_true=torch.from_numpy(x_true_band.transpose(0,2,1)).type(torch.FloatTensor)
y_true=torch.from_numpy(y_true).type(torch.LongTensor)
Label_true=Data.TensorDataset(x_true,y_true)

label_train_loader=Data.DataLoader(Label_train,batch_size=args.batch_size,shuffle=True)
label_test_loader=Data.DataLoader(Label_test,batch_size=args.batch_size,shuffle=True)
label_true_loader=Data.DataLoader(Label_true,batch_size=100,shuffle=False)


if args.flag_test == 'test':
    if args.mode == 'ViT':
        model.load_state_dict(torch.load('./ViT.pt'))      
    elif (args.mode == 'CAF') & (args.patches == 1):
        model.load_state_dict(torch.load('./SpectralFormer_pixel.pt'))
    elif (args.mode == 'CAF') & (args.patches == 7):
        model.load_state_dict(torch.load('./SpectralFormer_patch.pt'))
    else:
        raise ValueError("Wrong Parameters") 
    model.eval()
    tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
    OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)

    # output classification maps
    pre_u = test_epoch(model, label_true_loader, criterion, optimizer)
    prediction_matrix = np.zeros((height, width), dtype=float)
    for i in range(total_pos_true.shape[0]):
        prediction_matrix[total_pos_true[i,0], total_pos_true[i,1]] = pre_u[i] + 1
    plt.subplot(1,1,1)
    plt.imshow(prediction_matrix, colors.ListedColormap(color_matrix))
    plt.xticks([])
    plt.yticks([])
    plt.show()
    savemat('matrix.mat',{'P':prediction_matrix, 'label':label})
elif args.flag_test == 'train':
    print("start training")
    tic = time.time()
    for epoch in range(args.epoches): 
        scheduler.step()

        # train model
        model.train()
        train_acc, train_obj, tar_t, pre_t = train_epoch(model, label_train_loader, criterion, optimizer)
        OA1, AA_mean1, Kappa1, AA1 = output_metric(tar_t, pre_t) 
        print("Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f}"
                        .format(epoch+1, train_obj, train_acc))
        

        if (epoch % args.test_freq == 0) | (epoch == args.epoches - 1):         
            model.eval()
            tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
            OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
            print("Epoch: {:03d} val_OA: {:.4f}val_Kappa: {:.4f}".format(epoch+1, OA2, Kappa2))

    toc = time.time()
    print("Running Time: {:.2f}".format(toc-tic))
    print("**************************************************")

print("Final result:")
print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2))
print(AA2)
print("**************************************************")
print("Parameter:")

def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k,v))

print_args(vars(args))









