# Copyright 2022 Adobe. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
import math
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch import autograd

from pathlib import Path
file_path = Path(__file__).parent

from Util.arcface_pytorch.resnet_face_recognition import resnet_face18
from Util.landmark_util import Get_HeatMap_PyTorch

FACE_ID_LOSS_TYPE = ['MSE', 'CosineSimilarity']

def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss

def KD_loss(args, teacher_g, noise, inject_index, fake_img, fake_img_list, percept_loss, parsing_net):
    '''
    Usage:
        Define the l1 knowledge distillation loss + LPIPS loss
    '''

    fake_img_teacher_list = teacher_g(noise, return_rgb_list=True, inject_index=inject_index)
    fake_img_teacher = fake_img_teacher_list[-1]

    # Content-Aware Adjustment for fake_img and fake_img_teacher
    if parsing_net is not None:    
        teacher_img_parsing = Batch_Img_Parsing(fake_img_teacher, parsing_net, device)
        fake_img_teacher = Get_Masked_Tensor(fake_img_teacher, teacher_img_parsing, device, mask_grad=False)
        fake_img = Get_Masked_Tensor(fake_img, teacher_img_parsing, device, mask_grad=True)

    fake_img_teacher.requires_grad = True

    # kd_l1_loss
    if args.kd_mode == 'Output_Only':
        kd_l1_loss = args.kd_l1_lambda * torch.mean(torch.abs(fake_img_teacher - fake_img))
    elif args.kd_mode == 'Intermediate':
        for fake_img_teacher in fake_img_teacher_list:
            fake_img_teacher.requires_grad = True
        loss_list = [torch.mean(torch.abs(fake_img_teacher - fake_img)) for (fake_img_teacher, fake_img) in zip(fake_img_teacher_list, fake_img_list)] 
        kd_l1_loss = args.kd_l1_lambda * sum(loss_list)  


    # kd_lpips_loss
    if percept_loss is None:
        kd_lpips_loss = torch.tensor(0.0, device=device)
    else:
        if args.size > train_hyperparams.LPIPS_IMAGE_SIZE: # pooled the image for LPIPS for memory saving
            pooled_fake_img = Downsample_Image_256(fake_img)
            pooled_fake_img_teacher = Downsample_Image_256(fake_img_teacher)
            kd_lpips_loss = args.kd_lpips_lambda * torch.mean(percept_loss(pooled_fake_img, pooled_fake_img_teacher))

        else:
            kd_lpips_loss = args.kd_lpips_lambda * torch.mean(percept_loss(fake_img, fake_img_teacher))

    return kd_l1_loss, kd_lpips_loss

# ---------------------- L1 and LPIPS Loss ----------------------

def L1_Loss(output_tensor, target_tensor):
    '''
    Usage:
        A wrapper function to return l1 loss
    Args:
        output_tensor:  (torch.Tensor) of image in the shape of [N, 3, 256, 256] within range of [-1, 1]
        target_tensor:  (torch.Tensor) of image in the shape of [N, 3, 256, 256] within range of [-1, 1]    
    '''
    
    l1_loss = torch.mean(torch.abs(output_tensor - target_tensor))
    return l1_loss

def LPIPS_Loss(output_tensor, target_tensor, lpips_module):
    '''
    Usage:
        A wrapper function to return lpips perceptual loss
    Args:
        output_tensor:  (torch.Tensor) of image in the shape of [N, 3, 256, 256] within range of [-1, 1]
        target_tensor:  (torch.Tensor) of image in the shape of [N, 3, 256, 256] within range of [-1, 1]    
        lpips_module: (PerceptualLoss) of LPIPS    
    '''
    
    lpips_loss = torch.mean(lpips_module(output_tensor, target_tensor))
    return lpips_loss

# ---------------------- Face Identity Loss Related Functions ----------------------

def RGB_to_GrayScale(rgb_img):
    '''
    
    Usage:
        Convert a batch of RGB images to GrayScale images for face recognition network
    
    Args:
        rgb_img: (torch.Tensor) of images of shape [N,3,H,W] with range [-1, 1]
    '''
    
    RGB_to_GrayScale_Coef = [0.2989, 0.587, 0.114]
    gray_scale_img = 0
    for i in range(len(RGB_to_GrayScale_Coef)):
        gray_scale_img += RGB_to_GrayScale_Coef[i] * rgb_img[:, i:i+1, ...]
    
    return gray_scale_img

def Convert_Tensor_For_Face_Recognition_Loss(img_tensor):
    '''
    Usage:
        Convert a batch of image tensors for arcface recoginition network loss
    
    Args:
        img_tensor: (torch.Tensor) of images of shape [N,3,256,256] with range [-1, 1]
    
    Return:
        gray_scale_img_tensor_pooled: (torch.Tensor) of images of shape [N,1,128,128] with range [-1, 1]
    '''
    
    gray_scale_img_tensor = RGB_to_GrayScale(img_tensor)
    gray_scale_img_tensor_pooled = F.avg_pool2d(gray_scale_img_tensor, kernel_size = 2, stride = 2)
    return gray_scale_img_tensor_pooled

def Load_Face_Recognition_Network():
    '''
    Usage:
        Load a pretrained arcface face recognition network
    '''
    face_rec_model = resnet_face18(use_se = False)
    face_rec_model = nn.DataParallel(face_rec_model) # Data-Parallel it just for dictionary loading

    pretrained_model = (file_path / '''./arcface_pytorch/resnet18_arcfacenet.pth''').resolve()
    model_dict = torch.load(pretrained_model, map_location='cpu')
    face_rec_model.load_state_dict(model_dict)

    face_rec_model = face_rec_model.module # De-Parallel it
    return face_rec_model

def Face_Identity_Loss(output_tensor, target_tensor, face_rec_model, loss_type = 'MSE'):
    '''
    Usage:
        A wrapper function to return face identity loss
    Args:
        output_tensor:  (torch.Tensor) of image in the shape of [N, 3, 256, 256] within range of [-1, 1]
        target_tensor:  (torch.Tensor) of image in the shape of [N, 3, 256, 256] within range of [-1, 1]    
        face_rec_model: (torch.nn.Module) of the face recognition network
        loss_type:      (str) whether the loss is achieved by MSE or CosineSimialrity
    '''
    
    target_tensor_grayscale = Convert_Tensor_For_Face_Recognition_Loss(target_tensor)
    output_tensor_grayscale = Convert_Tensor_For_Face_Recognition_Loss(output_tensor)
    
    target_feature = face_rec_model(target_tensor_grayscale)
    output_feature = face_rec_model(output_tensor_grayscale)
    
    
    assert loss_type in FACE_ID_LOSS_TYPE
    if loss_type == 'MSE':
        loss = F.mse_loss(output_feature, target_feature)
    elif loss_type == 'CosineSimilarity': 
        loss = torch.mean(1 - F.cosine_similarity(output_feature, target_feature))
    return loss


# ---------------------- Heat Map Loss Related Functions ----------------------

def Heat_Map_Loss(g_output, r_input, fa, device):
    '''
    Usage:
        A face alginment loss based on the extracted heat maps of output images and render signals
    
    Args:
        g_output: (torch.Tensor) of output images of shape [N, 3, 256, 256] within the range of [-1, 1]
        r_input:  (torch.Tensor) of render input of shape [N, 3, 256, 256] within the range of [-1, 1]
        fa:       (FaceAlignment) defined in face_alignment.FaceAlignment
        device:   (str) name of the device to place the tensor
    '''
    
    heatmap_r, _, _ = Get_HeatMap_PyTorch(r_input, fa, device)
    heatmap_g, _, _ = Get_HeatMap_PyTorch(g_output, fa, device)
    
    heatmap_loss = torch.mean(torch.sum(torch.square(heatmap_r - heatmap_g), dim = (1,2,3)))
    return heatmap_loss



# ---------------------- Face Regional Loss Related Functions ----------------------

def Get_Render_Mask(render_img):
    '''
    Usage:
        Return a 2D mask of shape [N, 256, 256] on where the render image has values
    
    Args:
        render_img: (torch.Tensor) of shape [N, 3, 256, 256] where N is the batch size
    '''
    mask = torch.mean(render_img, dim = 1) > -1
    return mask.type(torch.FloatTensor)


def Face_Regional_Loss(r_img, g_img, device):
    '''
    Usage:
        Return the L2 loss on face region defined by the render face
    
    Args:
        r_img: (torch.Tensor) of shape [N, 3, 256, 256] of the render images
        g_img: (torch.Tensor) of shape [N, 3, 256, 256] of the generated images
    '''
    
    mask = Get_Render_Mask(r_img)
    mask_unsqueeze = mask.unsqueeze(1).to(device)
    masked_r = r_img * mask_unsqueeze
    masked_g = g_img * mask_unsqueeze

    face_regional_score = torch.mean(torch.square(masked_r - masked_g))
    return face_regional_score
