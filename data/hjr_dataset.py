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

DATA_PATH =""

class HJRDataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, imgtype, isdrone, sample_point, patch, band_patch, band
    ):         
        self.annotations = pd.read_csv(csv_file, encoding='UTF-8')        
        self.imgtype = imgtype
        self.isdrone = isdrone
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
            image, label = self.mat_open(path_img, self.imgtype, self.isdrone, self.sample_point, self.patch, self.band_patch, self.band)
        
        else:                    
            path_label = os.path.join(DATA_PATH, self.annotations.iloc[index, 1])       
            image = self.multiple_image_open(path_img, self.imgtype, self.sample_point, self.patch, self.band_patch, self.band)        
            label = self.label_open(path_label, self.imgtype, self.sample_point)
        
        
        out["image"] = image
        out["label"] = label
        

        return out

    def mat_open(self, path_img, imgtype, isdrone, sample_point, patch, band_patch, band):
         
        mat_data = loadmat(path_img)        
        image_mat = mat_data['image']       
   
        if not isdrone:
            if imgtype == 'RA':
                clipping = 40000
                image_mat = np.clip(image_mat.astype(np.float32)/clipping, 0.0, 1.0)                 
            elif imgtype =='RE':
                clipping = 65535
                image_mat = np.clip(image_mat.astype(np.float32)/clipping, 0.0, 1.0)                 
        else:
            clipping = 255
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
        self, csv_file, imgtype, isdrone, sample_point, patch, band_patch, band
    ):         
        self.annotations = pd.read_csv(csv_file, encoding='UTF-8')        
        self.imgtype = imgtype
        self.isdrone = isdrone
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
            try : 
                image, label, origin_image = self.mat_open(path_img, self.imgtype, self.isdrone, self.sample_point, self.patch, self.band_patch, self.band)
            except Exception as e:            
                with open('error.txt','a') as file:
                    file.write(path_img+'\n')
                print(path_img)
                raise e
        
        else:                    
            path_label = os.path.join(DATA_PATH, self.annotations.iloc[index, 1])       
            image, origin_image = self.multiple_image_open(path_img, self.imgtype, self.sample_point, self.patch, self.band_patch, self.band)        
            label = self.label_open(path_label, self.imgtype, self.sample_point)
        
        
        out["image"] = image
        out["label"] = label
        out["origin_image"] = origin_image
        out["path"] = path_img

        return out

    def mat_open(self, path_img, imgtype, isdrone, sample_point, patch, band_patch, band):
         
        try:
            mat_data = loadmat(path_img)
        except Exception as e:
            print(f'mat open error : {e}')
            raise e
            
        image_mat = mat_data['image']       
        
        if not isdrone:
            if imgtype == 'RA':
                clipping = 40000
                image_mat = np.clip(image_mat.astype(np.float32)/clipping, 0.0, 1.0)                 
            elif imgtype =='RE':
                clipping = 65535
                image_mat = np.clip(image_mat.astype(np.float32)/clipping, 0.0, 1.0)                 
        else:
            clipping = 255
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
        image_mat = image_mat[sample_point[:,1],sample_point[:,0]]
        label = torch.from_numpy(label_mat).type(torch.LongTensor)        
        
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
        your_variable_name = 'label'  

        
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
        
        self.imgtype = imgtype
        self.sample_point = sample_point
        self.patch = patch
        self.band_patch= band_patch
        self.band = band

        

    def __len__(self):
        return 1
        # return len(self.annotations)

    def __getitem__(self, index):
        out = dict()
        
        mat_folder = './input_data/temp/mat' 
        for root, dirs, files in os.walk(mat_folder):
            for image_filename in files:
                if image_filename.endswith('.mat'):
                    path_img = os.path.join(mat_folder, image_filename)

        
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
        image_mat = image_mat[sample_point[:,1],sample_point[:,0]]
        label = torch.from_numpy(label_mat).type(torch.LongTensor)        
        
                
        
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