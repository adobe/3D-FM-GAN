# Copyright 2022 Adobe. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
import argparse
import random
import os
import time
import datetime
from matplotlib import pyplot as plt
from easydict import EasyDict

import numpy as np
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms 

from stylegan2 import Generator, Discriminator
from resnet_encoder import resnet18
from psp_encoder_model.encoders import psp_encoders
from dataset import Synthetic_Dataset, FFHQ_Dataset_Reconstruction, FFHQ_Dataset_Editing, FFHQ_Dataset, DualSupervisionSampler, Data_Loading

from Miscellaneous.distributed import reduce_loss_dict  
from Util.network_util import Build_Generator_From_Dict, Forward_Inference
from Evaluation.quant_eval import Get_Recon_Score, Get_Edit_Score
from Evaluation.visual_eval import Get_Real_Img_Val_Sample, Get_Syn_Img_Val_Sample, Get_Batch_Eval_Result
from Evaluation.fid import load_patched_inception_v3
import lpips
from Util.training_util import d_logistic_loss, d_r1_loss, g_nonsaturating_loss, L1_Loss, LPIPS_Loss, Face_Identity_Loss, Load_Face_Recognition_Network, Heat_Map_Loss, Face_Regional_Loss

import sys
sys.path.append('/home/code-base/user_space/2021_work/third_party_code/face_alignment/')
import face_alignment

# Hyper-parameters for training!
import train_hyperparams

parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, default=train_hyperparams.data_folder)
parser.add_argument('--ffhq_train_img_dir', type=str, default=train_hyperparams.ffhq_train_img_dir)
parser.add_argument('--size', type=int, default=train_hyperparams.generated_img_size)
parser.add_argument('--ckpt', type=str, default=train_hyperparams.ckpt)
parser.add_argument('--channel_multiplier', type=int, default=train_hyperparams.channel_multiplier)
parser.add_argument('--latent', type=int, default=train_hyperparams.latent)
parser.add_argument('--n_mlp', type=int, default=train_hyperparams.n_mlp)
parser.add_argument('--mod_encode', type=str, default=train_hyperparams.mod_encode)
parser.add_argument('--mod_space', type=str, default=train_hyperparams.mod_space)
parser.add_argument('--mod_encoder_layer', type=int, default=train_hyperparams.mod_encoder_layer)
parser.add_argument('--w_plus_sliced_layer', type=list, default=train_hyperparams.w_plus_sliced_layer)
parser.add_argument('--co_mod', type=str, default=train_hyperparams.co_mod)
parser.add_argument('--use_tanh', type=bool, default=train_hyperparams.use_tanh)
parser.add_argument('--load_train_state', type=bool, default=train_hyperparams.load_train_state)

parser.add_argument('--iter', type=int, default=train_hyperparams.training_iters)
parser.add_argument('--rec_batch', type=int, default=train_hyperparams.rec_batch)
parser.add_argument('--rec_dataset_type', type=str, default=train_hyperparams.rec_dataset_type)
parser.add_argument('--ds_batch', type=int, default=train_hyperparams.ds_batch)
parser.add_argument('--ds_dataset_type', type=str, default=train_hyperparams.ds_dataset_type)
parser.add_argument('--ds_freq', type=int, default=train_hyperparams.ds_freq)
parser.add_argument('--lr', type=float, default=train_hyperparams.init_lr)

parser.add_argument('--d_reg_every', type=int, default=train_hyperparams.d_reg_freq)
parser.add_argument('--r1', type=float, default=train_hyperparams.discriminator_r1)
parser.add_argument('--use_g_reg', type=bool, default=train_hyperparams.use_g_reg)
parser.add_argument('--g_reg_every', type=int, default=train_hyperparams.g_reg_freq)
parser.add_argument('--path_reg_batch_shrink', type=int, default=train_hyperparams.path_reg_batch_shrink)
parser.add_argument('--generator_path_reg_weight', type=float, default=train_hyperparams.generator_path_reg_weight)

parser.add_argument('--l1_loss_lambda', type=float, default=train_hyperparams.l1_loss_lambda)
parser.add_argument('--lpips_loss_lambda', type=float, default=train_hyperparams.lpips_loss_lambda)
parser.add_argument('--face_id_loss_lambda', type=float, default=train_hyperparams.face_id_loss_lambda)
parser.add_argument('--face_id_loss_type', type=str, default=train_hyperparams.face_id_loss_type)
parser.add_argument('--hmap_loss_lambda', type=float, default=train_hyperparams.hmap_loss_lambda)
parser.add_argument('--hmap_iter_thres', type=int, default=train_hyperparams.hmap_iter_thres)
parser.add_argument('--face_reg_loss_lambda', type=float, default=train_hyperparams.face_reg_loss_lambda)

parser.add_argument('--n_real_eval_faces', type=int, default=train_hyperparams.n_real_eval_faces)
parser.add_argument('--n_syn_eval_faces', type=int, default=train_hyperparams.n_syn_eval_faces)
parser.add_argument('--visual_real_img_dir', type=str, default=train_hyperparams.visual_real_img_dir)
parser.add_argument('--val_sample_freq', type=int, default=train_hyperparams.val_sample_freq)
parser.add_argument('--model_save_freq', type=int, default=train_hyperparams.model_save_freq)
parser.add_argument('--quant_eval_img_dir', type=str, default=train_hyperparams.quant_eval_img_dir)
parser.add_argument('--quant_eval_batch_size', type=int, default=train_hyperparams.quant_eval_batch_size)

# Currently Not Used Params
parser.add_argument('--mixing', type=float, default=train_hyperparams.noise_mixing)

parser.add_argument('--fid_n_sample', type=int, default=train_hyperparams.fid_n_sample)
parser.add_argument('--fid_batch', type=int, default=train_hyperparams.fid_batch)

args = parser.parse_args()
n_gpu = len(train_hyperparams.gpu_device_ids)
device = train_hyperparams.primary_device
args.distributed = n_gpu > 1

def Print_Experiment_Status(exp_log_file):
    '''
    Usage:
        To print out all the relevant status of 
    '''
    experiment_status_str = '\n' + '--------------- Training Start ---------------' + '\n\n'
    experiment_status_str += 'Params: ' + '\n\n' + \
          '  Model and Training Data: ' + '\n' + \
          '    Synthetic Data Folder: ' + str(args.path) + '\n' + \
          '    FFHQ Data Folder: ' + str(args.ffhq_train_img_dir) + '\n' + \
          '    Generator Num Layers: ' + str(args.n_latent) + '\n' + \
          '    Latent Variable Dimension: ' + str(args.latent) + '\n' + \
          '    Generated Image Size: ' + str(args.size) + '\n' + \
          '    Channel Multiplier: ' + str(args.channel_multiplier) + '\n' + \
          '    Modulation Encoding: ' + str(args.mod_encode) + '\n' + \
          '    Modulation Space: ' + str(args.mod_space) + '\n' + \
          '    Num Modulation Encoder Layer: ' + str(args.mod_encoder_layer) + '\n' + \
          '    W+ Space Layer Slicing: ' + str(args.w_plus_sliced_layer) + '\n' + \
          '    Co-Modulation: ' + str(args.co_mod) + '\n' + \
          '    Tanh Clipping: ' + str(args.use_tanh) + '\n' + \
          '    Initial Checkpoint: ' + str(args.ckpt) + '\n' + \
          '    Load Training State: ' + str(args.load_train_state) + '\n\n' + \
          '  GPU Setup: ' + '\n' + \
          '    Distributed Training: ' + str(args.distributed) + '\n' + \
          '    Primiary GPU Device: ' + device + '\n' + \
          '    GPU Device IDs: ' + str(train_hyperparams.gpu_device_ids) + '\n' + \
          '    Number of GPUs: ' + str(n_gpu) + '\n\n' + \
          '  Training Params: ' + '\n' + \
          '    Training Iterations: ' + str(args.iter) + '\n' + \
          '    Reconstruction Dataset: ' + str(args.rec_dataset_type) + '\n' + \
          '    Reconstruction Batch Size: ' + str(args.rec_batch) + '\n' + \
          '    Dual Supervision Dataset: ' + str(args.ds_dataset_type) + '\n' + \
          '    Dual Supervision Batch Size: ' + str(args.ds_batch) + '\n' + \
          '    Dual Supervision Frequency: ' + str(args.ds_freq) + '\n' + \
          '    Learning Rate: ' + str(args.lr) + '\n' + \
          '    Use Generator Regularization: ' + str(args.use_g_reg) + '\n' + \
          '    Generator Regularization Frequency: ' + str(args.g_reg_every) + '\n' + \
          '    Generator Regularization Weight: ' + str(args.generator_path_reg_weight) + '\n' + \
          '    Generator Regularization Batch Shrink: ' + str(args.path_reg_batch_shrink) + '\n' + \
          '    Discriminator Regularization Frequency: ' + str(args.d_reg_every) + '\n' + \
          '    Discriminator Regularization Weight: ' + str(args.r1) + '\n' + \
          '    LPIPS Loss Weight: ' + str(args.lpips_loss_lambda) + '\n' + \
          '    L1 Loss Weight: ' + str(args.l1_loss_lambda) + '\n' + \
          '    Face ID Loss Weight: ' + str(args.face_id_loss_lambda) + '\n' + \
          '    Face ID Loss Type: ' + str(args.face_id_loss_type) + '\n' + \
          '    Heat Map Loss Weight: ' + str(args.hmap_loss_lambda) + '\n' + \
          '    Heat Map Loss Iteration Threshold: ' + str(args.hmap_iter_thres) + '\n' + \
          '    Face Regional Loss Weight: ' + str(args.face_reg_loss_lambda) + '\n\n' + \
          '  Validation Params: ' + '\n' + \
          '    Model Saving Frequency: ' + str(args.model_save_freq) + '\n' + \
          '    Quantitative Evaluation Directory: ' + str(args.quant_eval_img_dir) + '\n' + \
          '    Quantitative Evaluation Batch Size: ' + str(args.quant_eval_batch_size) + '\n' + \
          '    Visual Evaluation Frequency: ' + str(args.val_sample_freq) + '\n' + \
          '    Number of Visual Real Samples: ' + str(args.n_real_eval_faces) + '\n' + \
          '    Number of Visual Synthetic Samples: ' + str(args.n_syn_eval_faces) + '\n' + \
          '    Visual Real Samples Directory: ' + str(args.visual_real_img_dir) + '\n\n'

    print(experiment_status_str)
    exp_log_file.write(experiment_status_str)
    

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def Get_Readable_Cur_Time():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def Visual_Evaluation_Setup(args, synface_dataset, transform, device):
    '''
    Args:
        Set up the visual evaluation during training process
    '''
    real_img_list = [os.path.join(args.visual_real_img_dir, real_img) for real_img in os.listdir(args.visual_real_img_dir)]
    real_img_eval_samples = Get_Real_Img_Val_Sample(real_img_list, transform, args.n_real_eval_faces, device)
    syn_img_eval_samples = Get_Syn_Img_Val_Sample(synface_dataset, args.n_syn_eval_faces, device)
    visual_eval_samples = syn_img_eval_samples + real_img_eval_samples
    return visual_eval_samples


def Dataset_DataLoader_Setup(args):
    '''
    Usage:
        Set up the Dataset and DataLoader for training and evaluation
    '''

    # ============================== Building Dataset ==============================
    transform = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    ) # No need for random flipping for the synthetic dataset

    # Training Dataset
    print('Building Synface Dataset...')    
    synface_dataset = Synthetic_Dataset(args.path, transform)
    print('Dataset Built') 

    # Reconstruction Training Loader Building
    if args.rec_dataset_type == 'FFHQ':
        ffhq_recon_train_dataset = FFHQ_Dataset_Reconstruction(os.path.join(args.ffhq_train_img_dir, 'img'), 
                                                    os.path.join(args.ffhq_train_img_dir, 'render_img'), transform)  
        recon_dataset = ffhq_recon_train_dataset
    elif args.rec_dataset_type == 'Synthetic':
        recon_dataset = synface_dataset

    rec_train_loader = data.DataLoader(recon_dataset,
                                batch_size = args.rec_batch,
                                shuffle=True,
                                num_workers=8)

    # Dual Supervision Training Loader Building
    if args.ds_dataset_type == 'FFHQ':
        ffhq_edit_train_dataset = FFHQ_Dataset_Editing(os.path.join(args.ffhq_train_img_dir, 'img'), os.path.join(args.ffhq_train_img_dir, 'edit_render_img'), 
                                          transform, train=True, render_image_folder = os.path.join(args.ffhq_train_img_dir, 'render_img'))

        ds_train_loader = torch.utils.data.DataLoader(ffhq_edit_train_dataset,
                                          batch_size = args.ds_batch // 2,
                                          shuffle = True,
                                          num_workers = 8)

    elif args.ds_dataset_type == 'Synthetic':
        DS_sampler = DualSupervisionSampler(data_source = synface_dataset, n_img_per_id = 7)
        ds_train_loader = data.DataLoader(synface_dataset,
                                              batch_size = args.ds_batch,
                                              sampler = DS_sampler,
                                              num_workers = 8)

    # Pure FFHQ Loader for FFHQ Dual Supervision Training
    pure_ffhq_dataset = FFHQ_Dataset(os.path.join(args.ffhq_train_img_dir, 'img'), transform)
    pure_ffhq_loader = torch.utils.data.DataLoader(pure_ffhq_dataset,
                                          batch_size = args.ds_batch // 2,
                                          shuffle = True,
                                          num_workers = 8)


    # Reconstruction Evaluation Data Loader
    ffhq_rec_eval_dataset = FFHQ_Dataset_Reconstruction(os.path.join(args.quant_eval_img_dir, 'img'), 
                                                    os.path.join(args.quant_eval_img_dir, 'render_img'), transform)
    rec_eval_loader = data.DataLoader(ffhq_rec_eval_dataset,
                            batch_size = args.quant_eval_batch_size,
                            shuffle=False,
                            num_workers=8)

    # Editing Evaluation Data Loader
    ffhq_edit_eval_dataset = FFHQ_Dataset_Editing(os.path.join(args.quant_eval_img_dir, 'img'), 
                                                  os.path.join(args.quant_eval_img_dir, 'edit_render_img'), transform, train=False)
    edit_eval_loader = data.DataLoader(ffhq_edit_eval_dataset,
                            batch_size = args.quant_eval_batch_size // 4,
                            shuffle=False,
                            num_workers=8)

    return transform, synface_dataset, sample_data(rec_train_loader), sample_data(ds_train_loader), sample_data(pure_ffhq_loader), rec_eval_loader, edit_eval_loader


def Module_To_Train_Setup(args):
    '''
    Usage:
        Define the encoders, G & D that need to train
    '''

    # Initializing Modules
    if args.co_mod is None:
        E_Tsr = resnet18(tensor_encoding = True).to(device)

        if args.mod_space == 'W':
            E_W = resnet18(tensor_encoding = False).to(device)
        elif args.mod_space == 'W+':
            opts = EasyDict({'input_nc': 3, 'n_styles': int(math.log(args.size, 2)) * 2 - 2})  
            E_W = psp_encoders.GradualStyleEncoder(args.mod_encoder_layer, 'ir_se', opts).to(device)

    else:
        if args.co_mod == 'Tensor Transform':
            E_Tsr = resnet18(tensor_encoding = True, tensor_transform = True).to(device)
        else:
            E_Tsr = resnet18(tensor_encoding = False).to(device)
        opts = EasyDict({'input_nc': 3, 'n_styles': int(math.log(args.size, 2)) * 2 - 2})  
        E_W = psp_encoders.GradualStyleEncoder(args.mod_encoder_layer, 'ir_se', opts).to(device)

    D = Discriminator(args.size, channel_multiplier=args.channel_multiplier).to(device)

    if args.ds_dataset_type == 'FFHQ':
        D_ffhq = Discriminator(args.size, channel_multiplier=args.channel_multiplier).to(device)
    else:
        D_ffhq = None

    # If there is prior checkpoint
    ckpt = None        
    if args.ckpt is not None:    
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        G = Build_Generator_From_Dict(ckpt['g'], size=args.size, latent = args.latent).to(device)
        g_ema = Build_Generator_From_Dict(ckpt['g_ema'], size=args.size, latent = args.latent).to(device)
        D.load_state_dict(ckpt['d'])
        if ('e_tsr' in ckpt.keys()) and ('e_W' in ckpt.keys()):
            E_Tsr.load_state_dict(ckpt['e_tsr'], strict=False)
            E_W.load_state_dict(ckpt['e_W'], strict=False)
        if ('d_ffhq' in ckpt.keys()) and (ckpt['d_ffhq'] is not None):
            D_ffhq.load_state_dict(ckpt['d_ffhq'])
        
    else:
        G = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
        g_ema = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
        accumulate(g_ema, G, 0)

    g_ema.eval()
    args.n_latent = g_ema.n_latent

    if args.distributed:
        G = nn.DataParallel(G, device_ids=train_hyperparams.gpu_device_ids)
        D = nn.DataParallel(D, device_ids=train_hyperparams.gpu_device_ids)
        E_Tsr = nn.DataParallel(E_Tsr, device_ids=train_hyperparams.gpu_device_ids)
        E_W = nn.DataParallel(E_W, device_ids=train_hyperparams.gpu_device_ids)
        if D_ffhq is not None:            
            D_ffhq = nn.DataParallel(D_ffhq, device_ids=train_hyperparams.gpu_device_ids)

    return G, E_Tsr, E_W, D, D_ffhq, g_ema, ckpt


def Module_Fix_Setup(args):
    '''
    Usage:
        Set up fix modules of LPIPS, Face Recognition Network, Inception Network
    '''

    lpips_model = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True,
                             gpu_ids=[train_hyperparams.gpu_device_ids[0]])

    face_rec_model = Load_Face_Recognition_Network().to(device)
    requires_grad(face_rec_model, False)
    face_rec_model.eval()

    inception_model = load_patched_inception_v3().to(device)
    requires_grad(inception_model, False)
    inception_model.eval()

    fa_model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device = device)
    requires_grad(fa_model.face_detector.face_detector, False)
    fa_model.face_detector.face_detector.eval()
    requires_grad(fa_model.face_alignment_net, False)
    fa_model.face_alignment_net.eval()
        
    # Parallelize all Network Models if Applicable
    if args.distributed:
        face_rec_model = nn.DataParallel(face_rec_model, device_ids=train_hyperparams.gpu_device_ids)
        inception_model = nn.DataParallel(inception_model, device_ids=train_hyperparams.gpu_device_ids)
        fa_model.face_detector.face_detector = nn.DataParallel(fa_model.face_detector.face_detector, device_ids=train_hyperparams.gpu_device_ids)
        fa_model.face_alignment_net = nn.DataParallel(fa_model.face_alignment_net, device_ids=train_hyperparams.gpu_device_ids)
    return lpips_model, face_rec_model, inception_model, fa_model

def D_Loss_BackProp(G, E_Tsr, E_W, D, g_input, r_input, g_ref, args, device, loss_dict, d_optim, d_type = 'D'):
    '''
    Usage:
        To update the D based on the GAN loss
    '''

    requires_grad(G, False)
    requires_grad(E_Tsr, False)
    requires_grad(E_W, False)
    requires_grad(D, True)

    g_output = Forward_Inference(g_input, r_input, E_Tsr, E_W, G, args.mod_encode, args.co_mod, args.w_plus_sliced_layer, args.use_tanh)

    out_pred = D(g_output)
    ref_pred = D(g_ref)
    d_loss = d_logistic_loss(ref_pred, out_pred)

    if d_type == 'D':
        loss_dict['d'] = d_loss
        loss_dict['ref_score'] = ref_pred.mean()
        loss_dict['out_score'] = out_pred.mean()
    elif d_type == 'D_ffhq':
        loss_dict['d_ffhq'] = d_loss
        loss_dict['ref_score_ffhq'] = ref_pred.mean()
        loss_dict['out_score_ffhq'] = out_pred.mean()

    D.zero_grad()
    d_loss.backward()
    d_optim.step()

def D_Reg_BackProp(real_img, D, args, d_optim):
    '''
    Usage:
        To update the D based on the regularization
    '''

    real_img.requires_grad = True
    real_pred = D(real_img)
    r1_loss = d_r1_loss(real_pred, real_img)

    D.zero_grad()
    (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

    d_optim.step()
    return r1_loss

def G_Loss_BackProp(G, E_Tsr, E_W, D, g_input, r_input, g_ref, args, device, loss_dict, g_enc_optim, lpips_model, face_rec_model, fa_model, iter_idx):
    '''
    Usage:
        To update the G based on the GAN loss and KD loss
    '''

    requires_grad(G, True)
    requires_grad(E_Tsr, True)
    requires_grad(E_W, True)   
    requires_grad(D, False)

    g_output = Forward_Inference(g_input, r_input, E_Tsr, E_W, G, args.mod_encode, args.co_mod, args.w_plus_sliced_layer, args.use_tanh)

    # GAN Loss
    out_pred = D(g_output)
    g_loss = g_nonsaturating_loss(out_pred)
    loss_dict['g'] = g_loss

    lpips_loss = args.lpips_loss_lambda * LPIPS_Loss(g_output, g_ref, lpips_model)
    loss_dict['lpips'] = lpips_loss

    l1_loss = args.l1_loss_lambda * L1_Loss(g_output, g_ref)
    loss_dict['l1'] = l1_loss

    face_id_loss = args.face_id_loss_lambda * Face_Identity_Loss(g_output, g_ref, face_rec_model, args.face_id_loss_type)
    loss_dict['face_id'] = face_id_loss

    if iter_idx > args.hmap_iter_thres:
        hmap_loss = args.hmap_loss_lambda * Heat_Map_Loss(g_output, r_input, fa_model, device)
    else:
        hmap_loss = torch.tensor(0.0, device=device)
    loss_dict['hmap'] = hmap_loss

    face_reg_loss = args.face_reg_loss_lambda * Face_Regional_Loss(r_input, g_output, device)
    loss_dict['face_reg'] = face_reg_loss

    total_loss = g_loss + lpips_loss + l1_loss + face_id_loss + hmap_loss + face_reg_loss

    G.zero_grad()
    E_Tsr.zero_grad()
    E_W.zero_grad()

    total_loss.backward()
    g_enc_optim.step()


def G_Loss_BackProp_FFHQ_DS(G, E_Tsr, E_W, D_ffhq, g_input, r_edit_input, g_ref, args, device, loss_dict, g_enc_optim, face_rec_model):
    '''
    Usage:
        To update G and Encoders based on the GAN loss of FFHQ, and Identity Loss
    '''
    requires_grad(G, True)
    requires_grad(E_Tsr, True)
    requires_grad(E_W, True)   
    requires_grad(D_ffhq, False)

    g_output = Forward_Inference(g_input, r_edit_input, E_Tsr, E_W, G, args.mod_encode, args.co_mod, args.w_plus_sliced_layer, args.use_tanh)

    # GAN Loss
    out_pred = D_ffhq(g_output)
    g_loss = g_nonsaturating_loss(out_pred)
    loss_dict['g_ffhq'] = g_loss

    # Face ID Loss
    face_id_loss = args.face_id_loss_lambda * Face_Identity_Loss(g_output, g_ref, face_rec_model, args.face_id_loss_type)
    loss_dict['face_id_ffhq'] = face_id_loss

    total_loss = g_loss + face_id_loss

    G.zero_grad()
    E_Tsr.zero_grad()
    E_W.zero_grad()

    total_loss.backward()
    g_enc_optim.step()

    return g_output.detach()

def G_Reg_BackProp(G, E_Tsr, E_W, g_input, r_input, args, mean_path_length, g_enc_optim):
    '''
    Usage:
        To update the G, E_Tsr, and E_W based on the path regularization
    '''

    # Prepare the input 
    path_batch_size = max(1, args.rec_batch // args.path_reg_batch_shrink)
    rand_choice_idx = np.random.choice(range(args.rec_batch), size=path_batch_size, replace=False)
    g_input_reg, r_input_reg = g_input[rand_choice_idx, ...], r_input[rand_choice_idx, ...]

    # Input feed-forwarding    

    g_output, path_lengths = Forward_Inference(g_input_reg, r_input_reg, E_Tsr, E_W, G, args.mod_encode, args.co_mod, args.w_plus_sliced_layer, args.use_tanh, PPL_regularize=True)
    decay = 0.01
    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)
    path_loss = (path_lengths - path_mean).pow(2).mean()
    mean_path_length = path_mean.detach()

    G.zero_grad()
    E_Tsr.zero_grad()
    E_W.zero_grad()

    weighted_path_loss = args.generator_path_reg_weight * args.g_reg_every * path_loss

    if args.path_reg_batch_shrink:
        weighted_path_loss += 0 * g_output[0, 0, 0, 0]

    weighted_path_loss.backward()
    g_enc_optim.step()

    return path_loss, path_lengths, mean_path_length


def Training_Setup(exp_dir, g_ema, G, D, D_ffhq, E_Tsr, E_W, args):
    '''
    Usage:
        Setup the training and some key params
    '''
    sample_dir = exp_dir + '/sample/'
    ckpt_dir = exp_dir + '/ckpt/'
    os.mkdir(sample_dir)
    os.mkdir(ckpt_dir)
    g_ema_parallel = nn.DataParallel(g_ema, device_ids=train_hyperparams.gpu_device_ids)

    # Experiment Statistics Setup

    if args.distributed:
        g_module = G.module
        d_module = D.module
        e_tsr_module = E_Tsr.module
        e_W_module = E_W.module
        if D_ffhq is not None:
            d_ffhq_module = D_ffhq.module
        else:
            d_ffhq_module = None

    else:
        g_module = G
        d_module = D
        e_tsr_module = E_Tsr
        e_W_module = E_W
        d_ffhq_module = D_ffhq

    module_to_save = g_module, e_tsr_module, e_W_module, d_module, d_ffhq_module

    return sample_dir, ckpt_dir, g_ema_parallel, module_to_save


def Print_Train_Status(iter_idx, loss_dict, ds_flag, exp_log_file, time1, time3, args):
    '''
    Usage:
        Print the train status for an iteration
    '''
    loss_reduced = reduce_loss_dict(loss_dict)
    d_loss_val = loss_reduced['d'].mean().item()
    r1_val = loss_reduced['r1'].mean().item()
    path_loss_val = loss_reduced['g_reg'].mean().item()

    g_loss_val = loss_reduced['g'].mean().item()
    lpips_loss_val = loss_reduced['lpips'].mean().item()
    l1_loss_val = loss_reduced['l1'].mean().item()
    face_id_loss_val = loss_reduced['face_id'].mean().item()
    hmap_loss_val = loss_reduced['hmap'].mean().item()
    face_reg_loss_val = loss_reduced['face_reg'].mean().item()

    if (args.ds_dataset_type == 'Synthetic') or (ds_flag is False):

        exp_log_file.write('Iter #: ' + str(iter_idx) + ' Train Time: ' + str(round(time3 - time1, 2)) + ' Dual Supervision: ' + str(ds_flag) + ' D_Loss: ' + str(round(d_loss_val, 3))  + ' G_Loss: ' + str(round(g_loss_val, 3)) + ' G_Reg: ' + str(round(path_loss_val, 3)) + ' L1_Loss: ' + str(round(l1_loss_val, 3)) + ' LPIPS_Loss: ' + str(round(lpips_loss_val, 3)) + ' Face_ID_Loss: ' + str(round(face_id_loss_val, 3)) + ' Heat_Map_Loss: ' + str(round(hmap_loss_val, 3)) + ' Face_Regional_Loss: ' + str(round(face_reg_loss_val, 3)) + '\n'
        )

    elif (args.ds_dataset_type == 'FFHQ') and (ds_flag is True):
        g_loss_ffhq_val = loss_reduced['g_ffhq'].mean().item()
        d_loss_ffhq_val = loss_reduced['d_ffhq'].mean().item()
        face_id_loss_ffhq_val = loss_reduced['face_id_ffhq'].mean().item()

        exp_log_file.write('Iter #: ' + str(iter_idx) + ' Train Time: ' + str(round(time3 - time1, 2)) + ' Dual Supervision: ' + str(ds_flag) + ' D_Loss_FFHQ: ' + str(round(d_loss_ffhq_val, 3))  + ' G_Loss_FFHQ: ' + str(round(g_loss_ffhq_val, 3)) + ' Face_ID_Loss_FFHQ: ' + str(round(face_id_loss_ffhq_val, 3)) + ' D_Loss: ' + str(round(d_loss_val, 3))  + ' G_Loss: ' + str(round(g_loss_val, 3)) + ' G_Reg: ' + str(round(path_loss_val, 3)) + ' L1_Loss: ' + str(round(l1_loss_val, 3)) + ' LPIPS_Loss: ' + str(round(lpips_loss_val, 3)) + ' Face_ID_Loss: ' + str(round(face_id_loss_val, 3)) + '\n'
        )


def Sample_Eval_Save_Ckpt(iter_idx, args, g_ema_parallel, visual_eval_samples, eval_loaders, module_fix, sample_dir, ckpt_dir, module_to_save, optimizers, device, exp_log_file):
    '''
    Usage:
        Save the qualitative evaluation sample as well as the checkpoint
    '''

    rec_eval_loader, edit_eval_loader = eval_loaders
    g_module, e_tsr_module, e_W_module, d_module, d_ffhq_module = module_to_save  
    g_enc_optim, d_optim, d_ffhq_optim = optimizers
    lpips_model, face_rec_model, inception_model, fa_model = module_fix

    if iter_idx % args.val_sample_freq == 0:
        with torch.no_grad():
            g_ema_parallel.eval()
            e_tsr_module.eval()
            e_W_module.eval()
            generative_model = e_tsr_module, e_W_module, g_ema_parallel
            img_result_plot_list = Get_Batch_Eval_Result(visual_eval_samples, generative_model, mod_encode = args.mod_encode, co_modulation = args.co_mod, sliced_layer = args.w_plus_sliced_layer, use_tanh = args.use_tanh)

        n_row = args.n_real_eval_faces + args.n_syn_eval_faces
        n_col = 5
        wspace = 0.1
        hspace = 0.1
        
        dpi = 200
        fig_size = (n_col, n_row)
        
        plt.figure(dpi = dpi, figsize = fig_size)
        
        for i, img_np in enumerate(img_result_plot_list):
            plt.subplot(n_row, n_col, i + 1)
            plt.imshow(img_np)
            plt.axis('off')
        
        plt.subplots_adjust(wspace = wspace, hspace = hspace)
        save_img_file = os.path.join(sample_dir,  f'{str(iter_idx).zfill(6)}.png')
        plt.savefig(save_img_file)
        plt.close()

    if (iter_idx % args.model_save_freq == 0) and (iter_idx > 0):            
        with torch.no_grad():
            g_ema_parallel.eval()
            e_tsr_module.eval()
            e_W_module.eval()
            generative_model = e_tsr_module, e_W_module, g_ema_parallel
            eval_models = face_rec_model, lpips_model
            cos_score, lpips_score, l1_score = Get_Recon_Score(rec_eval_loader, device, generative_model, eval_models, mod_encode = args.mod_encode, co_modulation = args.co_mod, sliced_layer = args.w_plus_sliced_layer, use_tanh = args.use_tanh, info_print = False)
            eval_models = face_rec_model, inception_model, fa_model
            cos_score_edit, fid, hmap_score, lmark_score, face_reg_score = Get_Edit_Score(edit_eval_loader, device, generative_model, eval_models, mod_encode = args.mod_encode, co_modulation = args.co_mod, sliced_layer = args.w_plus_sliced_layer, use_tanh = args.use_tanh, info_print = False)

        exp_log_file.write('\n' + 'Reconstruction Quantitative Evaluation: '+ '\n' +
                           'Cosine Similarity: ' + str(cos_score) + '\n' + 
                           'LPIPS Score: ' + str(lpips_score) + '\n' +
                           'L1 Score: ' + str(l1_score) + '\n')

        exp_log_file.write('\n' + 'Edit Quantitative Evaluation: '+ '\n' +
                           'Edit Cosine Similarity: ' + str(cos_score_edit) + '\n' + 
                           'FID: ' + str(fid) + '\n' +
                           'Heat Map: ' + str(hmap_score) + '\n' +
                           'Land Mark: ' + str(lmark_score) + '\n' +
                           'Face Regional: ' + str(face_reg_score) + '\n\n')

        torch.save(
            {
                'g': g_module.state_dict(),
                'e_tsr': e_tsr_module.state_dict(),
                'e_W': e_W_module.state_dict(),
                'd': d_module.state_dict(),
                'd_ffhq': d_ffhq_module.state_dict() if d_ffhq_module is not None else None,
                'g_ema': g_ema_parallel.module.state_dict(),
                'g_enc_optim': g_enc_optim.state_dict(),
                'd_optim': d_optim.state_dict(),
                'd_ffhq_optim': d_ffhq_optim.state_dict() if d_ffhq_optim is not None else None,
                'mod_encode': args.mod_encode,
                'mod_space': args.mod_space,
                'co_mod': args.co_mod,
                'sliced_layer': args.w_plus_sliced_layer,
                'use_tanh': args.use_tanh,
            },
            ckpt_dir + f'{str(iter_idx).zfill(6)}.pt'
        )


def train(args, train_loaders, eval_loaders, module_to_train, optimizers, device, module_fix, visual_eval_samples,  exp_dir, exp_log_file):
    '''
    Usage:
        The main function for training the conditional GAN
    '''

    # Setup training procedure
    G, E_Tsr, E_W, D, D_ffhq, g_ema = module_to_train
    g_enc_optim, d_optim, d_ffhq_optim = optimizers
    lpips_model, face_rec_model, inception_model, fa_model = module_fix
    rec_train_loader, ds_train_loader, pure_ffhq_loader = train_loaders

    sample_dir, ckpt_dir, g_ema_parallel, module_to_save = Training_Setup(exp_dir, g_ema, G, D, D_ffhq, E_Tsr, E_W, args) 
    g_module = module_to_save[0] 

    r1_loss = torch.tensor(0.0, device=device)
    r1_loss_ffhq = torch.tensor(0.0, device=device)
    path_loss = torch.tensor(0.0, device=device)
    mean_path_length = 0
    loss_dict = {}
    accum = 0.5 ** (32 / (10 * 1000))


    for iter_idx in range(args.start_iter, args.iter):
        time1 = time.time()

        # Load data for learning 
        if (iter_idx % args.ds_freq) == (args.ds_freq - 1):
            ds_flag = True
        else:
            ds_flag = False

        # Check if it is loading data for self-supervised dual supervision
        if (args.ds_dataset_type == 'Synthetic') or (ds_flag is False): 
            ds_with_ffhq = False
            g_input, r_input, g_ref = Data_Loading(rec_train_loader, ds_train_loader, ds_flag, device)
        elif (args.ds_dataset_type == 'FFHQ') and (ds_flag is True):
            ds_with_ffhq = True
            g_input, r_input, r_edit_input, g_ref, ffhq_ref = Data_Loading(rec_train_loader, ds_train_loader, ds_flag, device, pure_ffhq_loader, args.ds_dataset_type) 
        
        # An extra step for FFHQ based dual supervision
        if ds_with_ffhq: 
            D_Loss_BackProp(G, E_Tsr, E_W, D_ffhq, g_input, r_edit_input, ffhq_ref, args, device, loss_dict, d_ffhq_optim, d_type = 'D_ffhq')
    
            # Discriminator regularization
            if iter_idx % args.d_reg_every == 0:
                r1_loss = D_Reg_BackProp(ffhq_ref, D_ffhq, args, d_ffhq_optim)
                
            loss_dict['r1_ffhq'] = r1_loss_ffhq

            # Use GAN loss to train the G 
            g_input = G_Loss_BackProp_FFHQ_DS(G, E_Tsr, E_W, D_ffhq, g_input, r_edit_input, g_ref, args, device, loss_dict, g_enc_optim, face_rec_model) 

        # Use GAN loss to train the D
        D_Loss_BackProp(G, E_Tsr, E_W, D, g_input, r_input, g_ref, args, device, loss_dict, d_optim)

        # Discriminator regularization
        if iter_idx % args.d_reg_every == 0:
            r1_loss = D_Reg_BackProp(g_ref, D, args, d_optim)

        loss_dict['r1'] = r1_loss

        # Use GAN loss to train the G 
        G_Loss_BackProp(G, E_Tsr, E_W, D, g_input, r_input, g_ref, args, device, loss_dict, g_enc_optim, lpips_model, face_rec_model, fa_model, iter_idx)

        # Generator regularization
        if (iter_idx % args.g_reg_every) == 0 and (args.use_g_reg):
            path_loss, path_lengths, mean_path_length = G_Reg_BackProp(G, E_Tsr, E_W, g_input, r_input, args, mean_path_length, g_enc_optim)
        loss_dict['g_reg'] = path_loss
            
        time3 = time.time()

        accumulate(g_ema, g_module, accum)

        # Print the training status 
        Print_Train_Status(iter_idx, loss_dict, ds_flag, exp_log_file, time1, time3, args)

        # Visualize some generated samples and save the checkpoints
        Sample_Eval_Save_Ckpt(iter_idx, args, g_ema_parallel, visual_eval_samples, eval_loaders, module_fix, sample_dir, ckpt_dir, module_to_save, optimizers, device, exp_log_file)
            

def main():
    # ============================== Dataset & DataLoader Setup ==============================

    transform, synface_dataset, rec_train_loader, ds_train_loader, pure_ffhq_loader, rec_eval_loader, edit_eval_loader = Dataset_DataLoader_Setup(args)

    # ============================== Building Network Model ==============================

    G, E_Tsr, E_W, D, D_ffhq, g_ema, ckpt = Module_To_Train_Setup(args)
    lpips_model, face_rec_model, inception_model, fa_model = Module_Fix_Setup(args)

    # ============================== Initializing Optimizers ==============================

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_enc_params = list(G.parameters()) + list(E_Tsr.parameters()) + \
                   list(E_W.parameters())
    g_enc_optim = optim.Adam(
        g_enc_params,
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        D.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )
    if args.ds_dataset_type == 'FFHQ':
        d_ffhq_optim = optim.Adam(
            D_ffhq.parameters(),
            lr=args.lr * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
        )
    else:
        d_ffhq_optim = None

    if args.load_train_state:
        g_enc_optim.load_state_dict(ckpt['g_enc_optim'])
        d_optim.load_state_dict(ckpt['d_optim'])
        args.start_iter = int(args.ckpt[-9: -3]) + 1
        if 'd_ffhq_optim' in ckpt.keys() and (D_ffhq is not None):
            d_ffhq_optim.load_state_dict(ckpt['d_ffhq_optim'])
    else:
        args.start_iter = 0

    # ============================== Training Start ==============================

    # Experiment Saving Directory
    cur_time = Get_Readable_Cur_Time()
    exp_dir = 'Exp_'+ cur_time
    os.mkdir(exp_dir)
    exp_log_file = open(exp_dir + '/' + cur_time + '_training_log.out', 'w')
    Print_Experiment_Status(exp_log_file)

    # Visual Evaluation Samples
    visual_eval_samples = Visual_Evaluation_Setup(args, synface_dataset, transform, device)

    # Start Training
    train_start_time = time.time()

    train_loaders = rec_train_loader, ds_train_loader, pure_ffhq_loader
    eval_loaders = rec_eval_loader, edit_eval_loader
    module_to_train = G, E_Tsr, E_W, D, D_ffhq, g_ema
    module_fix = lpips_model, face_rec_model, inception_model, fa_model
    optimizers = g_enc_optim, d_optim, d_ffhq_optim

    train(args, train_loaders, eval_loaders, module_to_train, optimizers, device, module_fix, visual_eval_samples, exp_dir, exp_log_file)
    train_end_time = time.time()

    exp_log_file.write('\n' + 'Total training time: ' + str(round(train_end_time - train_start_time, 3)))
    exp_log_file.close()


if __name__ == '__main__':
    main()
g_file)
    train_end_time = time.time()

    exp_log_file.write('\n' + 'Total training time: ' + str(round(train_end_time - train_start_time, 3)))
    exp_log_file.close()


if __name__ == '__main__':
    main()
