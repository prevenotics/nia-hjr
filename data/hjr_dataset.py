"""
Creates a Pytorch dataset to load the hjr dataset
"""

import torch
import os
import pandas as pd
from PIL import Image
import numpy as np
import sys, os
import tifffile as tiff
from scipy.io import loadmat
from utils.utils import make_output_directory, load_checkpoint_files, save_checkpoint, chooose_train_and_test_point, mirror_hsi, gain_neighborhood_pixel, gain_neighborhood_band, train_and_test_data, train_and_test_label, accuracy, output_metric, cal_results, get_point

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# from config_mf import CARBON_CLIPPING, SGRST_CLIPPING, label_mapping, DATA_PATH
# import cv2

DATA_PATH =""
CLIPPING = 40000

class HJRDataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, imgtype, sample_point, patch, band_patch, band
    ): 
        # self.annotations = pd.read_csv(csv_file, encoding='cp949')        
        self.annotations = pd.read_csv(csv_file, encoding='UTF-8')        
        self.imgtype = imgtype
        self.sample_point = sample_point
        # self.args = args
        self.patch = patch
        self.band_patch= band_patch
        self.band = band
        # self.img_dir = img_dir
        # self.label_dir = label_dir
        # self.transform = transform
        

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        out = dict()
        path_img = os.path.join(DATA_PATH, self.annotations.iloc[index, 0])
        _, ext = os.path.splitext(path_img)
        
        if ext =='.mat':
            image, label = self.mat_open(path_img, self.imgtype, self.sample_point, self.patch, self.band_patch, self.band)
        
        else:                    
            path_label = os.path.join(DATA_PATH, self.annotations.iloc[index, 1])       
            image = self.multiple_image_open(path_img, self.imgtype, self.sample_point, self.patch, self.band_patch, self.band)        
            label = self.label_open(path_label, self.imgtype, self.sample_point)
        
        
        out["image"] = image
        out["label"] = label

        return out

    def mat_open(self, path_img, imgtype, sample_point, patch, band_patch, band):
         
        mat_data = loadmat(path_img)
        
        image_mat = mat_data['image']       
        image_mat = np.clip(image_mat.astype(np.float32)/CLIPPING, 0.0, 1.0)                 
        data = np.zeros((sample_point.shape[0], patch, patch, band), dtype=float)
        for i in range(sample_point.shape[0]):
            data[i,:,:,:] = gain_neighborhood_pixel(image_mat, sample_point, i, patch)
        data = gain_neighborhood_band(data, band, band_patch, patch)
        image = torch.from_numpy(data.transpose(0, 2, 1)).type(torch.FloatTensor)
        
        label_mat = mat_data['label']
        label_mat = label_mat[sample_point[:,0],sample_point[:,1]]
        label = torch.from_numpy(label_mat).type(torch.LongTensor)
        
        
                
        
        return image, label

    def multiple_image_open(self, path_img, imgtype, sample_point, patch, band_patch, band):
        file_dir, file_name = os.path.split(path_img)
        base_name, file_extension = os.path.splitext(file_name)
        
        first_image_path = os.path.join(file_dir, f"{base_name}_{imgtype}01{file_extension}")
        first_image = tiff.imread(first_image_path)
        num_channels = first_image.shape[-1]
        image_shape = first_image.shape[:-1]
        concatenated_image = np.zeros(image_shape + (num_channels * 20,), dtype=np.float32)
        
        
        for i in range(1, 21):
            tif_file_path = os.path.join(file_dir, f"{base_name}_{imgtype}{i:02d}{file_extension}")
            if os.path.exists(tif_file_path):
                img = tiff.imread(tif_file_path)
                img = np.clip(img.astype(np.float32)/CLIPPING, 0.0, 1.0) 
                concatenated_image[..., (i - 1) * num_channels:i * num_channels] = img
        
        data = np.zeros((sample_point.shape[0], patch, patch, band), dtype=float)
        for i in range(sample_point.shape[0]):
            data[i,:,:,:] = gain_neighborhood_pixel(concatenated_image, sample_point, i, patch)
        data = gain_neighborhood_band(data, band, band_patch, patch)
        
        data = torch.from_numpy(data.transpose(0, 2, 1)).type(torch.FloatTensor)
        # concatenated_image = torch.from_numpy(concatenated_image.transpose(2, 0, 1)).float()  # From HWC to CHW
        return data
    
    def label_open(self, path_label, imgtype, sample_point):
        file_dir, file_name = os.path.split(path_label)
        base_name, file_extension = os.path.splitext(file_name)
        
        first_label_path = os.path.join(file_dir, f"{base_name}_{imgtype}01{file_extension}")
        
        label_mat = loadmat(first_label_path)
        your_variable_name = 'label'  # your variable name in the .mat file

        # 만약 label_mat의 크기가 512x512가 아니면 512x512로 만들어줍니다.
        desired_shape = (512, 512)

        if your_variable_name in label_mat:
            label_mat = label_mat[your_variable_name]

            if label_mat.shape != desired_shape:
                new_label_mat = np.zeros(desired_shape, dtype=label_mat.dtype)
                new_label_mat[:min(label_mat.shape[0], desired_shape[0]), :min(label_mat.shape[1], desired_shape[1])] = label_mat
                label_mat = new_label_mat
        
        label_mat = label_mat[sample_point[:,0],sample_point[:,1]]
        label_mat = torch.from_numpy(label_mat).type(torch.LongTensor)
        
        return label_mat
    
    
    
    
    
    
    
    # # def normalize_concat(self, i_i, i_s, l_c, l_t):
    # @classmethod
    # def normalize_concat(cls, i_i, i_s, l_c, l_t):
        
    #     i_i = np.array(i_i).astype(np.float32)/255.0
    #     ########if i_i == (512,512,4) in data50
    #     # i_i = i_i[...,0:3]
    #     #######################################
        
    #     i_s = np.array(i_s)
    #     i_s[np.isnan(i_s)] = 0
    #     # i_s = np.clip(np.array(i_s).astype(np.float32)/SGRST_CLIPPING, 0.0, 1.0) #임분고 영상 30이상은 클리핑
    #     i_s = np.clip(i_s.astype(np.float32)/SGRST_CLIPPING, 0.0, 1.0) #임분고 영상 30이상은 클리핑
    #     img = np.dstack((i_i,i_s))
    #     img = torch.from_numpy(img.transpose(2, 0, 1)).float()  # From HWC to CHW

        
    #     l_c = np.array(l_c)
    #     l_c[np.isnan(l_c)] = 0
        
    #     ####l_c 소수점버림##############################
    #     # l_c = np.trunc(l_c)
        
        
    #     # l_c = np.clip(np.array(l_c).astype(np.float32)/CARBON_CLIPPING, 0.0, 1.0)        
    #     # l_c = np.clip(l_c.astype(np.float32)/CARBON_CLIPPING, 0.0, 1.0)         
    #     l_c = np.clip((l_c.astype(np.float32)-CARBON_CLIPPING[0])/CARBON_CLIPPING[2], 0.0, 1.0).astype(np.float32)
    #     l_c = np.expand_dims(l_c, axis=-1)
    #     label_reg = torch.from_numpy(l_c.transpose(2, 0, 1))# From HWC to CHW

    #     l_t = np.array(l_t)
        
    #     ########if l_t == (512,512,2) in data50
    #     # l_t = l_t[...,0]
    #     # temp = np.full((512,512),255)
    #     # for k, v in label_mapping.items():
    #     #         temp[l_t == k] = v
    #     # l_t=temp                
    #     #######################################
        
    #     for k, v in label_mapping.items():
    #             l_t[l_t == k] = v
       
    #     l_t = np.expand_dims(l_t, axis=-1)
    #     label_cls = torch.from_numpy(l_t.transpose(2, 0, 1))# From HWC to CHW

    #     return img, label_cls, label_reg

    # # @classmethod
    # # def decode_target(cls, mask):
    # #     """decode semantic mask to RGB image"""
    # #     return cls.cmap[mask]