# Copyright 2022 Adobe. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
import torch
import torch.nn.functional as F
import numpy as np

from Util.training_util import Convert_Tensor_For_Face_Recognition_Loss, Get_Render_Mask
from Util.network_util import Forward_Inference, Forward_Inference_3_Encoder
from Util.landmark_util import Get_HeatMap_Landmark_PyTorch
from .fid import calc_fid

import pickle as pkl
from pathlib import Path
file_path = Path(__file__).parent

FFHQ_FID_STATS_FILE = str((file_path / '''./inception_ffhq_embed/ffhq_validation_256_inception_embeddings_eval_mode.pkl''').resolve())

def Compute_Face_Identity_Similarity(output_tensor, target_tensor, face_rec_model):
    '''
    Usage:
        Compute the cosine similarity between two recognition tensors
    Args:
        output_tensor:  (torch.Tensor) or (list) of (torch.Tensor) of image in the shape of [N, 3, 256, 256] within range of [-1, 1]
        target_tensor:  (torch.Tensor) of image in the shape of [N, 3, 256, 256] within range of [-1, 1]    
        face_rec_model: (torch.nn.Module) of the face recognition network
    '''
    with torch.no_grad():
        target_tensor_grayscale = Convert_Tensor_For_Face_Recognition_Loss(target_tensor)
        target_feature = face_rec_model(target_tensor_grayscale)

        if isinstance(output_tensor, list):
            cos_similarity = []
            for output_tensor_element in output_tensor:
                output_tensor_grayscale = Convert_Tensor_For_Face_Recognition_Loss(output_tensor_element)
                output_feature = face_rec_model(output_tensor_grayscale)
                cos_similarity.append(F.cosine_similarity(output_feature, target_feature))
            
        else:
            output_tensor_grayscale = Convert_Tensor_For_Face_Recognition_Loss(output_tensor)
            output_feature = face_rec_model(output_tensor_grayscale)
            cos_similarity = F.cosine_similarity(output_feature, target_feature)
    return cos_similarity

def Get_Recon_Score(eval_loader, device, generative_model, eval_models, info_print = False, **kwargs):
    '''
    Usage:
        Evaluate the model on its reconstruction ability
    
    Args:
        eval_loader:        (data.DataLoader) to load the evaluation data
        device:             (str) of the device 
        E_Tsr:              (nn.Module) of tensor encoder
        E_W:                (nn.Module) of W encoder
        g_ema:              (nn.DataParallel) module of StyleGAN2 generator
                            !!! For current inference path, g_ema MUST BE a nn.DataParallel Module!!!   
        face_rec_model:     (nn.Module) of the face recoginition network
        percept_loss:       (lpips.PerceptualLoss) of the LPIPS evaluation
        mod_encode:         (str) to determine what modulation encoder tries to encode
        co_modulation:      (bool) whether or not the inference is a co-modulation
        use_tanh:           (bool) whether or not to clip g_output within range of (-1, 1) by tanh
        info_print:         (bool) whether to print the running information or not
    '''

    if len(generative_model) == 3:  
        E_Tsr, E_W, g_ema = generative_model
        is_3_encoder = False
    elif len(generative_model) == 4:
        E_Tsr, E_W, E_W_Plus, g_ema =  generative_model
        is_3_encoder = True

    face_rec_model, percept_loss = eval_models

    cos_sim_list = []
    lpips_list = []
    l1_list = []
    
    with torch.no_grad():
        for idx, (p_input, r_input) in enumerate(eval_loader):
            if info_print:
                print('Batch: ' + str(idx))
            p_input = p_input.to(device)
            r_input = r_input.to(device)
            if is_3_encoder is False:
                g_output = Forward_Inference(p_input, r_input, E_Tsr, E_W, g_ema, **kwargs)
            else:
                g_output = Forward_Inference_3_Encoder(p_input, r_input, E_Tsr, E_W, E_W_Plus, g_ema, **kwargs)
            cos_sim_score = Compute_Face_Identity_Similarity(g_output, p_input, face_rec_model)
            cos_sim_list += cos_sim_score.cpu().tolist()

            lpips_score = percept_loss(g_output, p_input)
            lpips_list += lpips_score.cpu().tolist()

            l1_score = torch.mean(torch.abs(g_output - p_input), dim = (1,2,3))
            l1_list += l1_score.cpu().tolist()
    
    if info_print:
        print('Cosine Similarity Len: ' + (str(len(cos_sim_list))) + ' LPIPS Len: ' 
              + str(len(lpips_list)) + ' L1 Len: ' + str(len(l1_list)) )
    
    return np.mean(cos_sim_list), np.mean(lpips_list), np.mean(l1_list)


def Get_Edit_Score(eval_loader, device, generative_model, eval_models, info_print = False, **kwargs):
    '''
    Usage:
        Evaluate the model on its reconstruction ability
    
    Args:
        eval_loader:        (data.DataLoader) to load the evaluation data
        device:             (str) of the device 
        E_Tsr:              (nn.Module) of tensor encoder
        E_W:                (nn.Module) of W encoder
        g_ema:              (nn.DataParallel) module of StyleGAN2 generator
                            !!! For current inference path, g_ema MUST BE a nn.DataParallel Module!!!   
        face_rec_model:     (nn.Module) of the face recoginition network
        inception_model:    (nn.Module) of the inception-v3 classifier
        fa_model:           (FaceAlignment) defined in face_alignment.FaceAlignment
        mod_encode:         (str) to determine what modulation encoder tries to encode
        co_modulation:      (bool) whether or not the inference is a co-modulation
        use_tanh:           (bool) whether or not to clip g_output within range of (-1, 1) by tanh
        info_print:         (bool) whether to print the running information or not
    '''

    if len(generative_model) == 3:  
        E_Tsr, E_W, g_ema = generative_model
        is_3_encoder = False
    elif len(generative_model) == 4:
        E_Tsr, E_W, E_W_Plus, g_ema =  generative_model
        is_3_encoder = True

    face_rec_model, inception_model, fa_model = eval_models
    
    cos_sim_list = []
    hmap_score_list = []
    lmark_score_list = []
    face_diff_score_list = []
    incep_features = []

    with torch.no_grad():
        for idx,img_tensor_list in enumerate(eval_loader):
            if info_print:
                print('Batch: ' + str(idx))

            # Get Generated Images
            p_input = img_tensor_list[0]
            p_input = p_input.to(device)

            r_input_list = img_tensor_list[1:]
            g_output_list = []
            for r_input in r_input_list:
                r_input = r_input.to(device)
                if is_3_encoder is False:
                    g_output = Forward_Inference(p_input, r_input, E_Tsr, E_W, g_ema, **kwargs)
                else:
                    g_output = Forward_Inference_3_Encoder(p_input, r_input, E_Tsr, E_W, E_W_Plus, g_ema, **kwargs)

                g_output_list.append(g_output)

                # Face Regional Loss
                mask = Get_Render_Mask(r_input)
                mask_unsqueeze = mask.unsqueeze(1).to(device)
                masked_r = r_input * mask_unsqueeze
                masked_g = g_output * mask_unsqueeze
                
                face_diff_score = torch.mean(torch.square(masked_r - masked_g), dim = (1,2,3))
                face_diff_score_list += face_diff_score.cpu().tolist()

                # Heat Map & Landmark Loss
                heatmap_g, landmark_g = Get_HeatMap_Landmark_PyTorch(g_output, fa_model, device)
                heatmap_r, landmark_r = Get_HeatMap_Landmark_PyTorch(r_input, fa_model, device)

                heatmap_score = torch.sum(torch.square(heatmap_r - heatmap_g), dim = (1,2,3)).cpu().tolist()
                hmap_score_list += heatmap_score

                landmark_score = np.mean(np.square(landmark_r - landmark_g), axis = (1,2)).tolist()
                lmark_score_list += landmark_score


            # Cosine Similarity Loss
            cos_sim_score = Compute_Face_Identity_Similarity(g_output_list, p_input, face_rec_model)
            for cos_sim_element in cos_sim_score:
                cos_sim_list += cos_sim_element.cpu().tolist()
    
            # Inception Features
            g_output_tensor = torch.cat(g_output_list, 0)
            feat = inception_model(g_output_tensor)[0].view(g_output_tensor.shape[0], -1)
            incep_features.append(feat.to('cpu'))
    
     
    incep_features = torch.cat(incep_features, 0).cpu().numpy()
    sample_mean = np.mean(incep_features, 0)
    sample_cov = np.cov(incep_features, rowvar=False)
    
    inception_ffhq_stats = pkl.load(open(FFHQ_FID_STATS_FILE,'rb'))
    real_mean,real_cov = inception_ffhq_stats['mean'], inception_ffhq_stats['cov']
    fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)
    
    if info_print:
        print('Cosine Similarity Len: ' + (str(len(cos_sim_list))) + ' Inception Feature Shape: ' + str(incep_features.shape) +
              ' HeatMap Len: ' + str(len(hmap_score_list)) + ' LandMark Len: ' + str(len(lmark_score_list)) + ' FaceDiff Len: ' + str(len(face_diff_score_list)))
    
    return np.mean(cos_sim_list), fid, np.mean(hmap_score_list), np.mean(lmark_score_list), np.mean(face_diff_score_list)
