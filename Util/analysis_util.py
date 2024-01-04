# Copyright 2022 Adobe. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
import torch
from torch import nn
import numpy as np
from .network_util import Get_Layer_Output, Convert_Tensor_To_Image, Build_Generator_From_Dict
from resnet_encoder import resnet18
from models.encoders import psp_encoders
from easydict import EasyDict
import math

from matplotlib import pyplot as plt
import os

# ============================ New Model Log Parsing ============================

def Parse_A_Line(log_line):
    d_loss_str = 'D_Loss: '
    g_loss_str = 'G_Loss: '
    l1_loss_str = 'L1_Loss: '
    
    d_loss_start = log_line.find(d_loss_str) + len(d_loss_str)
    d_loss_end = log_line.find(g_loss_str)
    d_loss = float(log_line[d_loss_start : d_loss_end])
    
    g_loss_start = log_line.find(g_loss_str) + len(g_loss_str)
    g_loss_end = log_line.find(l1_loss_str)
    g_loss = float(log_line[g_loss_start : g_loss_end])

    return d_loss, g_loss

def Moving_Average(list_or_array):
    '''
    Usage: 
        Return the running mean of an np array
    Args:
        list_or_array: (list) or (array) to compute the running mean
    '''
    
    running_mean = [list_or_array[0]]
    for i in range(1, len(list_or_array)):
        base_mean = running_mean[i - 1]
        new_sample = list_or_array[i]
        new_sample_weight = 1 / (i + 1)
        added_mean = base_mean * (1 - new_sample_weight) + new_sample * new_sample_weight
        running_mean.append(added_mean)
    
    assert len(running_mean) == len(list_or_array)
    return running_mean


def Extract_Reconstruction_Evaluation_Score(exp_log_file):
    '''
    Usage:
        To extract the FID & FLOPs information from a training log
    Args:
        exp_dir: (str) of the checkpoint directory
    '''
    
    CosineSimilarity_STR = 'Cosine Similarity: '
    LPIPS_STR = 'LPIPS Score: '
    L1_STR = 'L1 Score: '

    CosineSimilarity_list = []
    LPIPS_list = []
    L1_list = []

    for line in open(exp_log_file, 'r').readlines():
        if (CosineSimilarity_STR in line) and ('Edit' not in line):
            CosineSimilarity = float(line[len(CosineSimilarity_STR):])
            CosineSimilarity_list.append(CosineSimilarity)

        elif LPIPS_STR in line:
            LPIPS = float(line[len(LPIPS_STR):])
            LPIPS_list.append(LPIPS)
            
        elif L1_STR in line:
            L1 = float(line[len(L1_STR):])
            L1_list.append(L1)            
    
    return CosineSimilarity_list, LPIPS_list, L1_list


def Extract_Edit_Evaluation_Score(exp_log_file):
    '''
    Usage:
        To extract the FID & FLOPs information from a training log
    Args:
        exp_dir: (str) of the checkpoint directory
    '''
    
    CosineSimilarity_STR = 'Edit Cosine Similarity: '
    FID_STR = 'FID: '
    HeatMap_STR = 'Heat Map: '
    LandMark_STR = 'Land Mark: '
    FaceRegional_STR = 'Face Regional:'

    CosineSimilarity_list = []
    FID_list = []
    HeatMap_list = []
    LandMark_list = []
    FaceReg_list = []

    for line in open(exp_log_file, 'r').readlines():
        if CosineSimilarity_STR in line:
            CosineSimilarity = float(line[len(CosineSimilarity_STR):])
            CosineSimilarity_list.append(CosineSimilarity)
            
        elif FID_STR in line:
            FID = float(line[len(FID_STR):])
            FID_list.append(FID)   

        elif HeatMap_STR in line:
            hmap = float(line[len(HeatMap_STR):])
            HeatMap_list.append(hmap)           

        elif LandMark_STR in line:
            lmark = float(line[len(LandMark_STR):])
            LandMark_list.append(lmark)   
    
        elif FaceRegional_STR in line:
            facereg = float(line[len(FaceRegional_STR):])
            FaceReg_list.append(facereg)
   
    return CosineSimilarity_list, FID_list, HeatMap_list, LandMark_list, FaceReg_list


def Model_Building_Func_2_Encoder(ckpt_model, device, gpu_device_ids):
    '''
    Usage:
        Build the inference model from the checkpoint
    '''
    
    # Specify the parameters
    mod_space = 'W'
    mod_encode = 'Render Image'
    co_mod = None

    if 'mod_space' in ckpt_model.keys():
        mod_space = ckpt_model['mod_space']

    if 'mod_encode' in ckpt_model.keys():
        mod_encode = ckpt_model['mod_encode']    

    if 'co_mod' in ckpt_model.keys():
        if ckpt_model['co_mod']  == True:
            co_mod = 'Multiplication'
        else:
            co_mod = ckpt_model['co_mod']

    if mod_space == 'W+':
        if len(ckpt_model['e_W'].keys()) == 325:
            wplus_mod_layer = 18 
        elif len(ckpt_model['e_W'].keys()) == 565:
            wplus_mod_layer = 50  

    size = 256
    latent_dim = 1024 if (co_mod == 'Concatenation') or (co_mod == 'Tensor Transform') else 512   
    

    # Build Generator
    g_ema = Build_Generator_From_Dict(ckpt_model['g_ema'], size=size, latent=latent_dim).to(device)
    g_ema.eval()
    g_ema = nn.DataParallel(g_ema, device_ids=gpu_device_ids)     
    
    
    if co_mod is None:
        E_Tsr = resnet18(tensor_encoding = True).to(device)

        if mod_space == 'W':
            E_W = resnet18(tensor_encoding = False).to(device)
        elif mod_space == 'W+':
            opts = EasyDict({'input_nc': 3, 'n_styles': int(math.log(size, 2)) * 2 - 2})  
            E_W = psp_encoders.GradualStyleEncoder(wplus_mod_layer, 'ir_se', opts).to(device)

    else:
        if co_mod == 'Tensor Transform':
            E_Tsr = resnet18(tensor_encoding = True, tensor_transform = True).to(device)
        else:
            E_Tsr = resnet18(tensor_encoding = False).to(device)
        opts = EasyDict({'input_nc': 3, 'n_styles': int(math.log(size, 2)) * 2 - 2})  
        E_W = psp_encoders.GradualStyleEncoder(wplus_mod_layer, 'ir_se', opts).to(device)


    E_Tsr.load_state_dict(ckpt_model['e_tsr'])
    E_W.load_state_dict(ckpt_model['e_W'])

    E_Tsr = nn.DataParallel(E_Tsr, device_ids = gpu_device_ids)
    E_W = nn.DataParallel(E_W, device_ids = gpu_device_ids)

    E_Tsr.eval()
    E_W.eval();
    
    return g_ema, E_Tsr, E_W, mod_encode, co_mod


def Model_Building_Func_3_Encoder(ckpt_model, device, gpu_device_ids):
    '''
    Usage:
        Build the inference model from the checkpoint
    '''
    
    # Specify the parameters

    if len(ckpt_model['e_W_Plus'].keys()) == 325:
        wplus_mod_layer = 18 
    elif len(ckpt_model['e_W_Plus'].keys()) == 565:
        wplus_mod_layer = 50  

    size = 256
    latent_dim = 512
    
    # Build Generator
    g_ema = Build_Generator_From_Dict(ckpt_model['g_ema'], size=size, latent=latent_dim).to(device)
    g_ema.eval()
    g_ema = nn.DataParallel(g_ema, device_ids=gpu_device_ids)     
    
    # Build Encoder
    E_Tsr = resnet18(tensor_encoding = True).to(device)
    E_W = resnet18(tensor_encoding = False).to(device)
    opts = EasyDict({'input_nc': 3, 'n_styles': int(math.log(size, 2)) * 2 - 2})  
    E_W_Plus = psp_encoders.GradualStyleEncoder(wplus_mod_layer, 'ir_se', opts).to(device)     

    E_Tsr.load_state_dict(ckpt_model['e_tsr'])
    E_W.load_state_dict(ckpt_model['e_W'])
    E_W_Plus.load_state_dict(ckpt_model['e_W_Plus'])

    E_Tsr = nn.DataParallel(E_Tsr, device_ids = gpu_device_ids)
    E_W = nn.DataParallel(E_W, device_ids = gpu_device_ids)
    E_W_Plus = nn.DataParallel(E_W_Plus, device_ids = gpu_device_ids)

    E_Tsr.eval()
    E_W.eval()
    E_W_Plus.eval()
    
    return g_ema, E_Tsr, E_W, E_W_Plus
