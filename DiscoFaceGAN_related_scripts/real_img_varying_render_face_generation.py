# Copyright 2022 Adobe. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
'''
This file will run with the code under DiscoFaceGAN environemnt
'''

import os
from PIL import Image

import tensorflow as tf
import numpy as np
import scipy.io
import time

import sys
sys.path.append('../DiscoFaceGAN/')
from generate_images import z_to_lambda_mapping, restore_weights_and_initialize
from renderer.face_decoder import Face3D
import dnnlib.tflib as tflib

tf_device = '/gpu:2'

tflib.init_tf()
with tf.device(tf_device):
    latents = tf.placeholder(tf.float32, name='latents', shape=[1,128+32+16+3])
    INPUTcoeff = z_to_lambda_mapping(latents)    
restore_weights_and_initialize()

with tf.device(tf_device):
    FaceRender = Face3D()
    INPUTcoeff_pl = tf.placeholder(tf.float32, name='input_coeff', shape=[1,254])
    INPUTcoeff_w_t = tf.concat([INPUTcoeff_pl, tf.zeros([1,3])], axis = 1)
    render_img,render_mask,render_landmark,_ = FaceRender.Reconstruction_Block(INPUTcoeff_w_t,256,1,progressive=False)  

data_root_dir = '/home/code-base/user_space/Dataset/FFHQ_DiscoFaceGAN_PreProcessed/Processed_File/train/'
coef_dir = os.path.join(data_root_dir, 'coeff')
coef_file_list = os.listdir(coef_dir)
edit_coef_save_dir = os.path.join(data_root_dir, 'edit_coeff')
edit_render_img_save_dir = os.path.join(data_root_dir, 'edit_render_img')

if not os.path.exists(edit_coef_save_dir):
    os.mkdir(edit_coef_save_dir)

if not os.path.exists(edit_render_img_save_dir):
    os.mkdir(edit_render_img_save_dir)

num_latent = 4
coef_dim = 254
identity_fixed_slice = [0, 160]

start_time = time.time()

for coef_file in coef_file_list:
    coef = scipy.io.loadmat(os.path.join(coef_dir, coef_file))['coeff']

    INPUTcoeff_list = []
    for _ in range(num_latent):
        latent_np = np.random.normal(size=[1,128+32+16+3])
        INPUTcoeff_np = tflib.run(INPUTcoeff, {latents: latent_np})
        INPUTcoeff_np[:,:160] = coef[:,:160]
        INPUTcoeff_list.append(INPUTcoeff_np)

    for idx, INPUT_coef_np in enumerate(INPUTcoeff_list):
        render_img_np = tflib.run(render_img, {INPUTcoeff_pl: INPUT_coef_np})
        render_img_pil = Image.fromarray(render_img_np[0].astype(np.uint8))
        render_img_file_name = 'r_edit_' +  coef_file.replace('.mat', '_' + str(idx) + '.png') 
        render_img_pil.save(os.path.join(edit_render_img_save_dir, render_img_file_name))

        coef_file_name = 'edit_' + coef_file.replace('.mat', '_' + str(idx) + '.npy')        
        np.save(os.path.join(edit_coef_save_dir, coef_file_name), INPUT_coef_np)

    print(coef_file)

end_time = time.time()
print('Total Running Time: ' + str(round(end_time - start_time, 2)))
