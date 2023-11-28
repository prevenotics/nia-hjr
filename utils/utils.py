import os
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from torchvision.transforms.functional import normalize
# from sklearn.metrics import plot_confusion_matrix

def load_checkpoint_files(ckpt_path, model, lr_scheduler, logger):
    print(f">>>>>>>>>>>>> Resuming training from checkpoint: {ckpt_path}")
    if logger:
        logger.info(f">>>>>>>>>>>>> Resuming training from checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model.load_state_dict(ckpt["model"])
    start_epoch = ckpt["epoch"] + 1
    if lr_scheduler:
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    best_loss = ckpt["best_loss"]
    best_acc = ckpt["best_acc"]

    del ckpt
    torch.cuda.empty_cache()

    return start_epoch, best_loss, best_acc

def save_checkpoint(model, optimizer, lr_scheduler, epoch, save_path,  best_loss=999, best_acc=-999):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'epoch': epoch,
                  'best_loss': best_loss,
                  'best_acc': best_acc}

    torch.save(save_state, save_path)


def make_directory(path):
    os.makedirs(path, exist_ok=True)

def make_output_directory(config):
    make_directory(config['path']['output_dir'])
    make_directory(config['test_param']['out_img_path'])
    make_directory(config['test_param']['out_npy_path'])
    
    # create log file
    log_dir = os.path.join(config['path']['output_dir'], "logs")
    make_directory(log_dir)

    # create tensorboard
    tensorb_dir = None
    if config['system']['tensorboard']:
        tensorb_dir = os.path.join(config['path']['output_dir'], "tensorb")
        make_directory(tensorb_dir)

    #ckpt
    pth_dir = os.path.join(config['path']['output_dir'], "pths")
    make_directory(pth_dir)

    return log_dir, pth_dir, tensorb_dir


#-------------------------------------------------------------------------------
# 定位训练和测试样本
def chooose_train_and_test_point(train_data, test_data, true_data, num_classes):
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}
    number_true = []
    pos_true = {}
    #-------------------------for train data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(train_data==(i+1))
        number_train.append(each_class.shape[0])
        pos_train[i] = each_class

    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]] #(695,2)
    total_pos_train = total_pos_train.astype(int)
    #--------------------------for test data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(test_data==(i+1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class

    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]] #(9671,2)
    total_pos_test = total_pos_test.astype(int)
    #--------------------------for true data------------------------------------
    for i in range(num_classes+1):
        each_class = []
        each_class = np.argwhere(true_data==i)
        number_true.append(each_class.shape[0])
        pos_true[i] = each_class

    total_pos_true = pos_true[0]
    for i in range(1, num_classes+1):
        total_pos_true = np.r_[total_pos_true, pos_true[i]]
    total_pos_true = total_pos_true.astype(int)

    return total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true
#-------------------------------------------------------------------------------
# 边界拓展：镜像
def mirror_hsi(height,width,band,input_normalize,patch=5):
    padding=patch//2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band),dtype=float)
    #中心区域
    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize
    #左边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=input_normalize[:,padding-i-1,:]
    #右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=input_normalize[:,width-1-i,:]
    #上边镜像
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]
    #下边镜像
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]

    # print("**************************************************")
    # print("patch is : {}".format(patch))
    # print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    # print("**************************************************")
    return mirror_hsi


def mirror(image, band, patch=5):
    height = width = image.shape[0]
    padding=patch//2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band),dtype=float)
    #中心区域
    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=image
    #左边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=image[:,padding-i-1,:]
    #右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=image[:,width-1-i,:]
    #上边镜像
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]
    #下边镜像
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]

    # print("**************************************************")
    # print("patch is : {}".format(patch))
    # print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    # print("**************************************************")
    return mirror_hsi
#-------------------------------------------------------------------------------
# 获取patch的图像数据
def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    half_patch = patch //2
    x = point[i,1]+half_patch
    y = point[i,0]+half_patch
    # temp_image = mirror_image[x:(x+patch),y:(y+patch),:]
    
    # if x-half_patch < 0 or y-half_patch < 0:        
    temp_image = mirror_image[x-half_patch:(x+half_patch+1),y-half_patch:(y+half_patch+1), :]
    # temp_image = mirror_image[:,:, x-half_patch:(x+half_patch+1),y-half_patch:(y+half_patch+1)]
    
    return temp_image

def gain_neighborhood_band(x_train, band, band_patch, patch=5):
    nn = band_patch // 2
    pp = (patch*patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch*patch, band)
    x_train_band = np.zeros((x_train.shape[0], patch*patch*band_patch, band),dtype=float)
    # 中心区域
    x_train_band[:,nn*patch*patch:(nn+1)*patch*patch,:] = x_train_reshape
    #左边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,:i+1] = x_train_reshape[:,:,band-i-1:]
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,i+1:] = x_train_reshape[:,:,:band-i-1]
        else:
            x_train_band[:,i:(i+1),:(nn-i)] = x_train_reshape[:,0:1,(band-nn+i):]
            x_train_band[:,i:(i+1),(nn-i):] = x_train_reshape[:,0:1,:(band-nn+i)]
    #右边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,:band-i-1] = x_train_reshape[:,:,i+1:]
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,band-i-1:] = x_train_reshape[:,:,:i+1]
        else:
            x_train_band[:,(nn+1+i):(nn+2+i),(band-i-1):] = x_train_reshape[:,0:1,:(i+1)]
            x_train_band[:,(nn+1+i):(nn+2+i),:(band-i-1)] = x_train_reshape[:,0:1,(i+1):]
    return x_train_band
#-------------------------------------------------------------------------------
# 汇总训练数据和测试数据
def train_and_test_data(mirror_image, band, train_point, test_point, true_point, patch=5, band_patch=3):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)
    x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype=float)
    for i in range(train_point.shape[0]):
        x_train[i,:,:,:] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    for j in range(test_point.shape[0]):
        x_test[j,:,:,:] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)
    for k in range(true_point.shape[0]):
        x_true[k,:,:,:] = gain_neighborhood_pixel(mirror_image, true_point, k, patch)
    print("x_train shape = {}, type = {}".format(x_train.shape,x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape,x_test.dtype))
    print("x_true  shape = {}, type = {}".format(x_true.shape,x_test.dtype))
    print("**************************************************")
    
    x_train_band = gain_neighborhood_band(x_train, band, band_patch, patch)
    x_test_band = gain_neighborhood_band(x_test, band, band_patch, patch)
    x_true_band = gain_neighborhood_band(x_true, band, band_patch, patch)
    print("x_train_band shape = {}, type = {}".format(x_train_band.shape,x_train_band.dtype))
    print("x_test_band  shape = {}, type = {}".format(x_test_band.shape,x_test_band.dtype))
    print("x_true_band  shape = {}, type = {}".format(x_true_band.shape,x_true_band.dtype))
    print("**************************************************")
    return x_train_band, x_test_band, x_true_band
#-------------------------------------------------------------------------------
# 标签y_train, y_test
def train_and_test_label(number_train, number_test, number_true, num_classes):
    y_train = []
    y_test = []
    y_true = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)
    for i in range(num_classes+1):
        for j in range(number_true[i]):
            y_true.append(i)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_true = np.array(y_true)
    print("y_train: shape = {} ,type = {}".format(y_train.shape,y_train.dtype))
    print("y_test: shape = {} ,type = {}".format(y_test.shape,y_test.dtype))
    print("y_true: shape = {} ,type = {}".format(y_true.shape,y_true.dtype))
    print("**************************************************")
    return y_train, y_test, y_true
#-------------------------------------------------------------------------------
def get_point(img_size, num):
    image_width = img_size
    image_height = img_size

    center_x = image_width // 2
    center_y = image_height // 2

    if num == 1:
        sampling_coords = np.array([[center_x, center_y]])
    else:
        sub_divisions = num+ 2
        x_coords = np.linspace(0, image_width, sub_divisions)
        y_coords = np.linspace(0, image_height, sub_divisions)

        x_mesh, y_mesh = np.meshgrid(x_coords[1:-1], y_coords[1:-1])
        sampling_coords = np.column_stack((x_mesh.ravel(), y_mesh.ravel())).astype(int)
    
    return sampling_coords
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def accuracy(output, target, topk=(1,), ignore_class=30):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  
  ignore_mask = (target != ignore_class).view(1, -1).expand_as(pred)
  ignore_cnt = target.shape[0] - torch.sum(ignore_mask).item()
  correct = pred.eq(target.view(1, -1).expand_as(pred))
  correct = correct * ignore_mask  # ignore_class를 제외한 부분에만 1이 되도록 마스크 적용
#   correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)    
    res.append(correct_k.mul_(100.0/(batch_size)))   
        
  return res, target, pred.squeeze(), ignore_cnt
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def output_metric(tar, pre, ignore_class=30):
    # matrix = confusion_matrix(tar, pre, labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29, 30])
    # matrix = matrix[:30,:30] #ignore_class = 30
    # OA, Kappa = cal_results(matrix)
    tar_no_ignore = tar[tar !=ignore_class]
    pre_no_ignore = pre[tar !=ignore_class]
    matrix = confusion_matrix(tar_no_ignore, pre_no_ignore, labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29])
    # OA, AA_mean, Kappa, AA = cal_results(matrix)
    
    ####################################################################################################################################################################
    # labels=["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31",]
    # plot_confusion_matrix(matrix, labels = labels)
    ####################################################################################################################################################################
    # plt.savefig("cm.jpg")
    OA, Kappa = cal_results(matrix)
    # return OA, AA_mean, Kappa, AA
    return OA, Kappa #, matrix
#-------------------------------------------------------------------------------

def output_metric_with_savefig(tar, pre, path, ignore_class=30):
    # matrix = confusion_matrix(tar, pre, labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29, 30])
    # matrix = matrix[:30,:30] #ignore_class = 30
    # OA, Kappa = cal_results(matrix)
    tar_no_ignore = tar[tar !=ignore_class]
    pre_no_ignore = pre[tar !=ignore_class]
    matrix = confusion_matrix(tar_no_ignore, pre_no_ignore, labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29])
    # OA, AA_mean, Kappa, AA = cal_results(matrix)
    
    ####################################################################################################################################################################
    labels=["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30"]
    plot_confusion_matrix(matrix, labels, path)
    ###################################################################################################################################################################
    
    OA, Kappa = cal_results(matrix)
    # return OA, AA_mean, Kappa, AA
    return OA, Kappa #, matrix
#-------------------------------------------------------------------------------

def cal_results(matrix):
    epsilon = 1e-15
    shape = np.shape(matrix)
    number = 0
    sum = 0
    # AA = np.zeros([shape[0]], dtype=np.float)
    for i in range(shape[0]): 
        number += matrix[i, i]
        # try:
        #     # AA[i] = matrix[i, i] / (np.sum(matrix[i, :]) + epsilon)
        #     AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        # except ZeroDivisionError:
        #     AA[i] = 0
        #     print("!!!!!!!!!!!!!!!!!!!!zero division!!!!!!!!!!!!!!!!!!!!")
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    # AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    # return OA, AA_mean, Kappa, AA
    return OA, Kappa
#-------------------------------------------------------------------------------

# confusion matrix 그리는 함수 
def plot_confusion_matrix(con_mat, labels, path, title='Confusion Matrix', cmap=plt.cm.get_cmap('Blues'), normalize=False):
    plt.figure(figsize=[25,25])
    plt.imshow(con_mat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    marks = np.arange(len(labels))
    nlabels = []
    for k in range(len(con_mat)):
        n = sum(con_mat[k])
        nlabel = '{0}(n={1})'.format(labels[k],n)
        nlabels.append(nlabel)
    plt.xticks(marks, labels)
    plt.yticks(marks, nlabels)

    thresh = con_mat.max() / 2.
    if normalize:
        for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
            plt.text(j, i, '{0}%'.format(con_mat[i, j] * 100 / n), horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")
    else:
        for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
            plt.text(j, i, con_mat[i, j], horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(path, 'cm.png'))
    plt.clf()
    



#DeepLabV3plus utils    
def denormalize(tensor, mean, std):
    mean = np.array(mean)
    std = np.array(std)

    _mean = -mean/std
    _std = 1/std
    return normalize(tensor, _mean, _std)

class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)

def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum

def fix_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def class_name():
    cls_name = [
        '01.갈파래류',
        '02.청각류',
        '03.대마디말류',
        '04.그물바탕말류',
        '05.모자반류',
        '06.나래미역류',
        '07.감태류',
        '08.유절산호말류',
        '09.무절산호말류',
        '10.우뭇가사리류',
        '11.도박류',
        '12.돌가사리류',
        '13.새우말류',
        '14.거머리말류',
        '15.암반류',
        '16.모래류',
        '17.인공어초류',
        '18.성게류',
        '19.불가사리류',
        '20.소라류',
        '21.군소전복류',
        '22.해면류',
        '23.담치류',
        '24.따개비류',
        '25.고둥류',
        '26.군부류',
        '27.조개류',
        '28.연성경성산호류',
        '29.해양쓰레기류',
        '30.폐어구류',
        '31.기타'        
    ]    
    return cls_name