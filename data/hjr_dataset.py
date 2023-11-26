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
from utils.utils import make_output_directory, load_checkpoint_files, save_checkpoint, chooose_train_and_test_point, mirror, gain_neighborhood_pixel, gain_neighborhood_band, train_and_test_data, train_and_test_label, accuracy, output_metric, cal_results, get_point
import utils.utils as util
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# from config_mf import CARBON_CLIPPING, SGRST_CLIPPING, label_mapping, DATA_PATH
# import cv2

DATA_PATH =""
# CLIPPING = 40000

class HJRDataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, imgtype, sample_point, patch, band_patch, band
    ): 
        # self.annotations = pd.read_csv(csv_file, encoding='cp949')        
        self.annotations = pd.read_csv(csv_file, encoding='UTF-8')        
        self.imgtype = imgtype
        self.sample_point = sample_point
        self.patch = patch
        self.band_patch= band_patch
        self.band = band
        
        

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
        
        if imgtype == 'RA':
            clipping = 40000
            image_mat = np.clip(image_mat.astype(np.float32)/clipping, 0.0, 1.0)                 
        elif imgtype =='RE':
            clipping = 65535
            image_mat = np.clip(image_mat.astype(np.float32)/clipping, 0.0, 1.0)                 
        data = np.zeros((sample_point.shape[0], patch, patch, band), dtype=float)
        if patch > 1:
            image_mat_mirror = mirror(image_mat, band, patch)        
            for i in range(sample_point.shape[0]):
                data[i,:,:,:] = gain_neighborhood_pixel(image_mat_mirror, sample_point, i, patch)        
        else:
            for i in range(sample_point.shape[0]):
                data[i,:,:,:] = gain_neighborhood_pixel(image_mat, sample_point, i, patch)        
        data = gain_neighborhood_band(data, band, band_patch, patch)
        image = torch.from_numpy(data.transpose(0, 2, 1)).type(torch.FloatTensor)
        
        label_mat = mat_data['label']
        label_mat = label_mat[sample_point[:,1],sample_point[:,0]]
        label = torch.from_numpy(label_mat).type(torch.LongTensor)
        
        # image_mat = torch.from_numpy(image_mat.transpose(0, 2, 1)).type(torch.FloatTensor)
        
                
        
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
                if imgtype == 'RA':
                    clipping = 40000
                    image_mat = np.clip(image_mat.astype(np.float32)/clipping, 0.0, 1.0)                 
                elif imgtype =='RE':
                    clipping = 65535
                    image_mat = np.clip(image_mat.astype(np.float32)/clipping, 0.0, 1.0)      
                concatenated_image[..., (i - 1) * num_channels:i * num_channels] = img
        
        data = np.zeros((sample_point.shape[0], patch, patch, band), dtype=float)
        for i in range(sample_point.shape[0]):
            data[i,:,:,:] = gain_neighborhood_pixel(concatenated_image, sample_point, i, patch)
        data = gain_neighborhood_band(data, band, band_patch, patch)
        
        data = torch.from_numpy(data.transpose(0, 2, 1)).type(torch.FloatTensor)
        # concatenated_image = torch.from_numpy(concatenated_image.transpose(0, 2, 1)).type(torch.FloatTensor)
        
        return data
    
    def label_open(self, path_label, imgtype, sample_point):
        file_dir, file_name = os.path.split(path_label)
        base_name, file_extension = os.path.splitext(file_name)
        
        first_label_path = os.path.join(file_dir, f"{base_name}_{imgtype}01{file_extension}")
        
        label_mat = loadmat(first_label_path)
        your_variable_name = 'label'  # your variable name in the .mat file
        
        desired_shape = (512, 512)

        if your_variable_name in label_mat:
            label_mat = label_mat[your_variable_name]

            if label_mat.shape != desired_shape:
                new_label_mat = np.zeros(desired_shape, dtype=label_mat.dtype)
                new_label_mat[:min(label_mat.shape[0], desired_shape[0]), :min(label_mat.shape[1], desired_shape[1])] = label_mat
                label_mat = new_label_mat
        
        label_mat = label_mat[sample_point[:,1],sample_point[:,0]]
        label_mat = torch.from_numpy(label_mat).type(torch.LongTensor)
        
        return label_mat
    
    
    
class HJRDataset_for_test(torch.utils.data.Dataset):
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
            image, label, origin_image = self.mat_open(path_img, self.imgtype, self.sample_point, self.patch, self.band_patch, self.band)
        
        else:                    
            path_label = os.path.join(DATA_PATH, self.annotations.iloc[index, 1])       
            image, origin_image = self.multiple_image_open(path_img, self.imgtype, self.sample_point, self.patch, self.band_patch, self.band)        
            label = self.label_open(path_label, self.imgtype, self.sample_point)
        
        
        out["image"] = image
        out["label"] = label
        out["origin_image"] = origin_image
        out["path"] = path_img

        return out

    def mat_open(self, path_img, imgtype, sample_point, patch, band_patch, band):
         
        mat_data = loadmat(path_img)
        basename = os.path.basename(path_img)
        image_mat = mat_data['image']       
        
        if basename[2]=='L' or basename[2]=='U':
            if image_mat.shape[0] != 512:
                return 0
        else:
            if image_mat.shape[0] != 256:
                return 0
            
        if imgtype == 'RA':
            clipping = 40000
            image_mat = np.clip(image_mat.astype(np.float32)/clipping, 0.0, 1.0)                 
        elif imgtype =='RE':
            clipping = 65535
            image_mat = np.clip(image_mat.astype(np.float32)/clipping, 0.0, 1.0)      
            
            
            
        
        
        # for i in range(batch_size):
        #         temp = origin_image[i,:,:].reshape(image_size, image_size, band)[:,:,[15,39,80]]
                
        # temp = (image_mat[:,:,[15,39,80]]*255).astype(np.uint8)       
        
                     
        data = np.zeros((sample_point.shape[0], patch, patch, band), dtype=float)

        if patch > 1:
            image_mat_mirror = mirror(image_mat, band, patch)        
            for i in range(sample_point.shape[0]):
                data[i,:,:,:] = gain_neighborhood_pixel(image_mat_mirror, sample_point, i, patch)        
        else:
            for i in range(sample_point.shape[0]):
                data[i,:,:,:] = gain_neighborhood_pixel(image_mat, sample_point, i, patch)        
        
        data = gain_neighborhood_band(data, band, band_patch, patch)
        image = torch.from_numpy(data.transpose(0, 2, 1)).type(torch.FloatTensor)
        
        label_mat = mat_data['label']
        # temp_label = label_mat*8
        label_mat = label_mat[sample_point[:,1],sample_point[:,0]]
        # temp_label_after = np.array(label_mat).reshape(256,256)*8
        # temp_image = image_mat[:,:,[15,39,80]]
        # res_image = image_mat[:, :, [80, 39, 15]]
        image_mat = image_mat[sample_point[:,1],sample_point[:,0]]
        # temp_image_after = image_mat.reshape(256,256,100)[:,:,[15,39,80]]
        
        
        label = torch.from_numpy(label_mat).type(torch.LongTensor)        
        # image_mat = torch.from_numpy(image_mat.transpose(0, 2, 1)).type(torch.FloatTensor)
        
                
        
        return image, label, image_mat

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
                if imgtype == 'RA':
                    clipping = 40000
                    image_mat = np.clip(image_mat.astype(np.float32)/clipping, 0.0, 1.0)                 
                elif imgtype =='RE':
                    clipping = 65535
                    image_mat = np.clip(image_mat.astype(np.float32)/clipping, 0.0, 1.0)      
                concatenated_image[..., (i - 1) * num_channels:i * num_channels] = img
        
        data = np.zeros((sample_point.shape[0], patch, patch, band), dtype=float)
        for i in range(sample_point.shape[0]):
            data[i,:,:,:] = gain_neighborhood_pixel(concatenated_image, sample_point, i, patch)
        data = gain_neighborhood_band(data, band, band_patch, patch)
        
        data = torch.from_numpy(data.transpose(0, 2, 1)).type(torch.FloatTensor)
        # concatenated_image = torch.from_numpy(concatenated_image.transpose(0, 2, 1)).type(torch.FloatTensor)
        
        return data, concatenated_image
    
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
        
        label_mat = label_mat[sample_point[:,1],sample_point[:,0]]
        label_mat = torch.from_numpy(label_mat).type(torch.LongTensor)
        
        return label_mat
    
    
    
class HJRDataset_for_online(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, imgtype, sample_point, patch, band_patch, band
    ): 
        # self.annotations = pd.read_csv(csv_file, encoding='cp949')        
        # self.annotations = pd.read_csv(csv_file, encoding='UTF-8')        
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
        return 1
        # return len(self.annotations)

    def __getitem__(self, index):
        out = dict()
        
        mat_folder = './input_data/temp/mat' # To 설 : temp가 영상이 들어올때마다 새롭게 생성해서 변경해주면 될듯
        for root, dirs, files in os.walk(mat_folder):
            # if "02.수중 및 지상 초분광" in root:
            for image_filename in files:
                if image_filename.endswith('.mat'):
                    # base_name, file_extension = os.path.splitext(image_filename)
                    path_img = os.path.join(mat_folder, image_filename)
        # if base_name.split('_')[:-1] =='E':
        #     self.imgtype = 'RE'
        # elif base_name.split('_')[:-1] =='A':
        #     self.imgtype = 'RA'
        # prefix = '_'.join(image_filename.split('_')[:-1])  # Extract the common prefix
        
        # if prefix = 'L':
        #     sample_point = util.get_point(256, cfg['test_param']['sampling_num'])   
        # if prefix == 'L' or prefix == 'U':
        #     self.band = 100
        # else:
        #     self.band = 80
            
        
        _, ext = os.path.splitext(path_img)
        
        if ext =='.mat':
            image, label, origin_image = self.mat_open(path_img, self.imgtype, self.sample_point, self.patch, self.band_patch, self.band)
        
        else:                    
            path_label = os.path.join(DATA_PATH, self.annotations.iloc[index, 1])       
            image, origin_image = self.multiple_image_open(path_img, self.imgtype, self.sample_point, self.patch, self.band_patch, self.band)        
            label = self.label_open(path_label, self.imgtype, self.sample_point)
        
        
        out["image"] = image
        out["label"] = label
        out["origin_image"] = origin_image
        out["path"] = path_img

        return out

    def mat_open(self, path_img, imgtype, sample_point, patch, band_patch, band):
         
        mat_data = loadmat(path_img)
        
        image_mat = mat_data['image']       
        
        if imgtype == 'RA':
            clipping = 40000
            image_mat = np.clip(image_mat.astype(np.float32)/clipping, 0.0, 1.0)                 
        elif imgtype =='RE':
            clipping = 65535
            image_mat = np.clip(image_mat.astype(np.float32)/clipping, 0.0, 1.0)      
            
            
            
        
        
        # for i in range(batch_size):
        #         temp = origin_image[i,:,:].reshape(image_size, image_size, band)[:,:,[15,39,80]]
                
        # temp = (image_mat[:,:,[15,39,80]]*255).astype(np.uint8)       
        
                     
        data = np.zeros((sample_point.shape[0], patch, patch, band), dtype=float)

        if patch > 1:
            image_mat_mirror = mirror(image_mat, band, patch)        
            for i in range(sample_point.shape[0]):
                data[i,:,:,:] = gain_neighborhood_pixel(image_mat_mirror, sample_point, i, patch)        
        else:
            for i in range(sample_point.shape[0]):
                data[i,:,:,:] = gain_neighborhood_pixel(image_mat, sample_point, i, patch)        
        
        data = gain_neighborhood_band(data, band, band_patch, patch)
        image = torch.from_numpy(data.transpose(0, 2, 1)).type(torch.FloatTensor)
        
        label_mat = mat_data['label']
        # temp_label = label_mat*8
        label_mat = label_mat[sample_point[:,1],sample_point[:,0]]
        # temp_label_after = np.array(label_mat).reshape(256,256)*8
        # temp_image = image_mat[:,:,[15,39,80]]
        image_mat = image_mat[sample_point[:,1],sample_point[:,0]]
        # temp_image_after = image_mat.reshape(256,256,100)[:,:,[15,39,80]]
        
        
        label = torch.from_numpy(label_mat).type(torch.LongTensor)        
        # image_mat = torch.from_numpy(image_mat.transpose(0, 2, 1)).type(torch.FloatTensor)
        
                
        
        return image, label, image_mat

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
                if imgtype == 'RA':
                    clipping = 40000
                    image_mat = np.clip(image_mat.astype(np.float32)/clipping, 0.0, 1.0)                 
                elif imgtype =='RE':
                    clipping = 65535
                    image_mat = np.clip(image_mat.astype(np.float32)/clipping, 0.0, 1.0)      
                concatenated_image[..., (i - 1) * num_channels:i * num_channels] = img
        
        data = np.zeros((sample_point.shape[0], patch, patch, band), dtype=float)
        for i in range(sample_point.shape[0]):
            data[i,:,:,:] = gain_neighborhood_pixel(concatenated_image, sample_point, i, patch)
        data = gain_neighborhood_band(data, band, band_patch, patch)
        
        data = torch.from_numpy(data.transpose(0, 2, 1)).type(torch.FloatTensor)
        # concatenated_image = torch.from_numpy(concatenated_image.transpose(0, 2, 1)).type(torch.FloatTensor)
        
        return data, concatenated_image
    
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
        
        label_mat = label_mat[sample_point[:,1],sample_point[:,0]]
        label_mat = torch.from_numpy(label_mat).type(torch.LongTensor)
        
        return label_mat