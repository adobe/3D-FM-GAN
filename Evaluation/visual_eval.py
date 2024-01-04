# Copyright 2022 Adobe. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
from PIL import Image
import imageio
import numpy as np
import random
import torch

from Util.network_util import Forward_Inference, Forward_Inference_3_Encoder

from copy import deepcopy
import os

VAL_SET_LEN = 3
NUM_IMG_PER_ID = 7

def tensor2im(image_tensor, imtype=np.uint8, cent=1., factor=255./2.):
    '''
    Usage:
        Transforming a PyTorch tensor from [-1, 1] to a numpy image of [0, 255]
    Args:
        image_tensor: (torch.Tensor) of range [-1, 1]
        imtype:       (datatype) default as np.uint8
        cent:         (float) of offset
        factor:       (float) of scale
    '''
    
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.clip(image_numpy, a_min = -1, a_max = 1)
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + cent) * factor
    return image_numpy.astype(imtype)

def Get_Real_Img_Val_Sample(real_img_val_list, transform, num_faces, device):
    '''
    Usage:
        Return real image and its rendered faces for model visual validation
    Args:
        real_img_val_list: (list) of the file paths of the (real image, rendered faces) tuple
        num_faces:         (int) of number of test faces
    '''
    
    sel_real_img_file = np.random.choice(real_img_val_list, size = num_faces, replace = False)
    real_img_val_sets = []
    for img_file in sel_real_img_file:
        img_np = list(np.load(img_file))
        test_set = img_np[:2] + [random.choice(img_np[2:])]
        torch_test_set = [transform(Image.fromarray(img)).unsqueeze(dim = 0).to(device) for img in test_set]
        real_img_val_sets += torch_test_set
    return real_img_val_sets


def Get_Syn_Img_Val_Sample(synface_dataset, num_faces, device):
    '''
    Usage:
        Return synthetic image and its rendered faces for model visual validation
    Args:
        synface_dataset: (data.Dataset) of self-built synthetic dataset
        num_faces:       (int) of number of test faces
    '''
    
    num_id = len(synface_dataset) // NUM_IMG_PER_ID
    load_idx = []
    for person_id in np.random.choice(num_id, num_faces):
        idx = person_id * NUM_IMG_PER_ID + np.random.choice(NUM_IMG_PER_ID, num_faces)
        load_idx += list(idx)

    syn_img_val_sets = []
    for i,idx in enumerate(load_idx):
        g_img, r_img = synface_dataset[idx]
        if i % 2 == 0:
            syn_img_val_sets += [g_img.unsqueeze(dim = 0).to(device), 
                                    r_img.unsqueeze(dim = 0).to(device)]
        else:
            syn_img_val_sets += [r_img.unsqueeze(dim = 0).to(device)]        
    
    return syn_img_val_sets


def Get_Single_Eval_Result(g_input, r_input_list, generative_model, **kwargs):
    '''
    Usage:
        Return a list of images for sample visualization
    Args:
        g_input:            (torch.Tensor) normalized image to be encoded for GAN's tensor
        r_input_list:       (list) of (torch.Tensor) normalized rendered faces to be encoded for W
        E_Tsr:              (nn.Module) of tensor encoder
        E_W:                (nn.Module) of W encoder
        g_ema:              (nn.DataParallel) module of StyleGAN2 generator
                            !!! For current inference path, g_ema MUST BE a nn.DataParallel Module!!!
        mod_encode:         (str) to determine what modulation encoder tries to encode
        co_modulation:      (bool) whether or not the inference is a co-modulation
        use_tanh:           (bool) whether or not to clip g_output within range of (-1, 1) by tanh
    '''

    if len(generative_model) == 3:  
        E_Tsr, E_W, g_ema = generative_model
        is_3_encoder = False
    elif len(generative_model) == 4:
        E_Tsr, E_W, E_W_Plus, g_ema =  generative_model
        is_3_encoder = True
    
    img_plot_list = [tensor2im(g_input)]
    with torch.no_grad():
        for r_input in r_input_list:
            img_plot_list.append(tensor2im(r_input))
            if is_3_encoder is False:
                g_output = Forward_Inference(g_input, r_input, E_Tsr, E_W, g_ema, **kwargs)
            else:
                g_output = Forward_Inference_3_Encoder(g_input, r_input, E_Tsr, E_W, E_W_Plus, g_ema, **kwargs)
            img_plot_list.append(tensor2im(g_output))
    return img_plot_list

def Get_Batch_Eval_Result(eval_img_set, generative_model, **kwargs):
    '''
    Usage:
        Return all samples' visualization result at once
    
    args:
        eval_img_set:       (list) of images for testing
        E_Tsr:              (nn.Module) of the tensor encoder for testing
        E_W:                (nn.Module) of the modulation encoder for testing 
        g_ema:              (nn.Module) of the generator for testing
        mod_encode:         (str) to determine what modulation encoder tries to encode
        use_tanh:           (bool) whether or not to clip g_output within range of (-1, 1) by tanh
    '''
    num_test_set = len(eval_img_set) // VAL_SET_LEN
    img_result_plot_list = []
    for i in range(num_test_set):
        test_set = eval_img_set[i * VAL_SET_LEN : (i + 1) * VAL_SET_LEN]
        g_input = test_set[0]
        r_input_list = test_set[1:]
        img_result = Get_Single_Eval_Result(g_input, r_input_list, generative_model, **kwargs)
        img_result_plot_list += img_result
    
    return img_result_plot_list 


# =========================== GIF Style Generative Evaluation ===========================

def Get_Single_Photo_Multi_Render_Result(photo_img_png_path, render_img_gif_path, model_modules, transform, device, **kwargs):
    '''
    Usage:
        Get the evaluation results for single photo image and multiple render images forwarding
    Args:
        photo_img_png_path:  (str) of the path of the single frame photo image ending with .png
        render_img_gif_path: (str) of the path of the multi-frame render images ending with .gif
        model_modules:       (list) of the modules of g_ema, e_tsr, e_W, mod_encode for evaluation
        device:              (str) of the device to place the tensor on 
        is_3_encoder:        (bool) whether the inference is based on 3 encooder architecture
    '''

    # Load Single Photo Image for Input
    p_input = transform(Image.open(photo_img_png_path)).to(device)
    p_input = p_input.unsqueeze(0)
    
    # Load a Series of Render Image from GIF for Input
    r_input_list = Load_GIF_As_Img_List(render_img_gif_path, transform)
        
    # Result Evaluation
    if len(model_modules) == 3:  
        E_Tsr, E_W, g_ema = model_modules
        is_3_encoder = False
    elif len(model_modules) == 4:
        E_Tsr, E_W, E_W_Plus, g_ema =  model_modules
        is_3_encoder = True
    
    with torch.no_grad():
        output_img_list = []
        for r_input in r_input_list:
            r_input = r_input.to(device)
            r_input = r_input.unsqueeze(0)
            if is_3_encoder:
                g_output = Forward_Inference_3_Encoder(p_input, r_input, E_Tsr, E_W, E_W_Plus, g_ema, **kwargs)
            else:
                g_output = Forward_Inference(p_input, r_input, E_Tsr, E_W, g_ema, **kwargs)
            out_img = tensor2im(g_output)
            output_img_list.append(out_img)  
        
    return output_img_list

def Load_GIF_As_Img_List(gif_path, transform):
    '''
    Usage:
        Load a GIF file as a list of transformed PyTorch tensors
    
    Args:
        gif_path:  (str) of the path to the gif image
        transform: (torchvision.transforms) to transform a PIL image
    '''
    gif_img = Image.open(gif_path)
    tensor_list = []
    for i in range(gif_img.n_frames):
        gif_img.seek(i)
        frame = deepcopy(gif_img)
        tensor = transform(frame.convert('RGB'))
        tensor_list.append(tensor)
    
    return tensor_list

def Get_Img_Path_Single_Factor_Test(test_dir):
    '''
    Usage:
        Get pairs of (photo_img_path, render_img_path) from a directory
    
    Args:
        test_dir: (str) of the path to the directory containing test images
    '''

    photo_img_files = [file for file in os.listdir(test_dir) if '.png' in file]
    gif_img_files = [file for file in os.listdir(test_dir) if '.gif' in file]
    
    test_list = []
    for gif_file in gif_img_files:
        for photo_file in photo_img_files:
            if photo_file[:-4] in gif_file:
                test_list.append((photo_file, gif_file))
    
    return test_list

def Test_Single_Factor_Editing(test_dir, save_dir, model_name, model_modules, transform, device, **kwargs):
    '''
    Usage:
        Conduct a visualization test for a single factor editing
    
    Args:
        test_dir:      (str) of the path to the directory containing test images
        save_dir:      (str) of the saving root for the result 
        model_modules: (list) of the modules of g_ema, e_tsr, e_W for evaluation
        model_name:    (str) of model name to distinguish result saving
        transform:     (torchvision.transforms) to transform a PIL image
        device:        (str) of the device to place the tensor on 
    '''
    
    test_pair_list = Get_Img_Path_Single_Factor_Test(test_dir)
    for test_pair in test_pair_list:
        photo_img_png_path = os.path.join(test_dir, test_pair[0])
        render_img_gif_path = os.path.join(test_dir, test_pair[1])
        output_img_list = Get_Single_Photo_Multi_Render_Result(photo_img_png_path, render_img_gif_path, 
                                                               model_modules, transform, device, **kwargs)
        result_file_name = model_name + '_result_' + test_pair[1]
        result_file_path = os.path.join(save_dir, result_file_name)
        imageio.mimsave(result_file_path, output_img_list)

def Get_Img_Path_Video_Test(test_dir):
    '''
    Usage:
        Get pairs of (photo_img_path, render_img_path) from a video directory
    
    Args:
        test_dir: (str) of the path to the directory containing test images
    '''

    photo_img_file = [file for file in os.listdir(test_dir) if '.png' in file][0]
    test_list = [(photo_img_file, 'render_img.gif'), ('photo_img.gif', 'render_img.gif')]
    
    return test_list

def Get_Multi_Photo_Multi_Render_Result(photo_img_gif_path, render_img_gif_path, model_modules, transform, device, **kwargs):
    '''
    Usage:
        Get the evaluation results for single photo image and multiple render images forwarding
    Args:
        photo_img_gif_path:  (str) of the path of the multi-frame photo images ending with .gif
        render_img_gif_path: (str) of the path of the multi-frame render images ending with .gif
        model_modules:       (list) of the modules of g_ema, e_tsr, e_W, mod_encode for evaluation
        device:              (str) of the device to place the tensor on 
    '''

    # Load a Series of Photo Images from GIF for Input
    p_input_list = Load_GIF_As_Img_List(photo_img_gif_path, transform)
    
    # Load a Series of Render Image from GIF for Input
    r_input_list = Load_GIF_As_Img_List(render_img_gif_path, transform)
        
    # Result Evaluation
    if len(model_modules) == 3:  
        E_Tsr, E_W, g_ema = model_modules
        is_3_encoder = False
    elif len(model_modules) == 4:
        E_Tsr, E_W, E_W_Plus, g_ema =  model_modules
        is_3_encoder = True
    
    with torch.no_grad():
        output_img_list = []
        for p_input, r_input in zip(p_input_list, r_input_list):
            r_input = r_input.unsqueeze(0).to(device)
            p_input = p_input.unsqueeze(0).to(device)
            if is_3_encoder:
                g_output = Forward_Inference_3_Encoder(p_input, r_input, E_Tsr, E_W, E_W_Plus, g_ema, **kwargs)
            else:
                g_output = Forward_Inference(p_input, r_input, E_Tsr, E_W, g_ema, **kwargs)
            out_img = tensor2im(g_output)
            output_img_list.append(out_img)  
        
    return output_img_list

def Test_Video_Reconstruction_Reanimation(test_dir, save_dir, video_name, model_name, model_modules, transform, device, **kwargs):
    '''
    Usage:
        Conduct a visualization test for a single factor editing
    
    Args:
        test_dir:      (str) of the path to the directory containing test images
        save_dir:      (str) of the saving root for the result 
        video_name:    (str) of the video name
        model_modules: (list) of the modules of g_ema, e_tsr, e_W for evaluation
        model_name:    (str) of model name to distinguish result saving
        transform:     (torchvision.transforms) to transform a PIL image
        device:        (str) of the device to place the tensor on 
    '''
    
    
    test_pair_list = Get_Img_Path_Video_Test(test_dir)
    
    # Reanimation Test
    photo_img_png_path = os.path.join(test_dir, test_pair_list[0][0])
    render_img_gif_path = os.path.join(test_dir, test_pair_list[0][1])
    output_img_list = Get_Single_Photo_Multi_Render_Result(photo_img_png_path, render_img_gif_path, 
                                                               model_modules, transform, device, **kwargs)
    
    result_file_name = model_name + '_' + video_name + '_reanimation_' + test_pair_list[0][0].replace('.png', '.gif')
    result_file_path = os.path.join(save_dir, result_file_name)
    imageio.mimsave(result_file_path, output_img_list)
    
    # Reconstruction Test
    photo_img_gif_path = os.path.join(test_dir, test_pair_list[1][0])
    render_img_gif_path = os.path.join(test_dir, test_pair_list[1][1])
    output_img_list = Get_Multi_Photo_Multi_Render_Result(photo_img_gif_path, render_img_gif_path, 
                                                               model_modules, transform, device, **kwargs)
        
    result_file_name = model_name + '_' + video_name + '_reconstruction_' + test_pair_list[1][0]
    result_file_path = os.path.join(save_dir, result_file_name)
    imageio.mimsave(result_file_path, output_img_list)
