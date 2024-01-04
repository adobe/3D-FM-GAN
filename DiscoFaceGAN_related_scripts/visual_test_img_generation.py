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
import random
import scipy.io

import sys
sys.path.append('../DiscoFaceGAN/')
from generate_images import load_Gs, z_to_lambda_mapping, truncate_generation, restore_weights_and_initialize
from renderer.face_decoder import Face3D
import dnnlib.tflib as tflib

tf_device = '/gpu:2'

tflib.init_tf()
with tf.device(tf_device):
    latents = tf.placeholder(tf.float32, name='latents', shape=[1,128+32+16+3])
    noise = tf.placeholder(tf.float32, name='noise', shape=[1,32])
    INPUTcoeff = z_to_lambda_mapping(latents)    
restore_weights_and_initialize()

with tf.device(tf_device):
    FaceRender = Face3D()
    INPUTcoeff_pl = tf.placeholder(tf.float32, name='input_coeff', shape=[1,254])
    INPUTcoeff_w_t = tf.concat([INPUTcoeff_pl, tf.zeros([1,3])], axis = 1)
    render_img,render_mask,render_landmark,_ = FaceRender.Reconstruction_Block(INPUTcoeff_w_t,256,1,progressive=False)  


num_latent = 4
coef_dim = 254
identity_fixed_slice = [0, 160]
num_test_imgs = 500
rand_img_chosen_list = np.random.choice(aligned_img_list, size = num_test_imgs, replace = False)
save_img_dir = '/home/code-base/user_space/Dataset/Self_Constructed_Visual_Val_Imgs_3DGAN/'
if not os.path.exists(save_img_dir):
    os.mkdir(save_img_dir)

for idx, img_chosen in enumerate(rand_img_chosen_list):
    coef_chosen = img_chosen.replace('png', 'mat')
    img_file = os.path.join(Aligned_Data_Path, img_chosen)
    coef_file = os.path.join(Aligned_Data_Coeff_Path, coef_chosen)
    coef = scipy.io.loadmat(coef_file)['coeff']

    INPUTcoeff_list = [coef[:,:coef_dim]]
    for _ in range(num_latent):
        latent_np = np.random.normal(size=[1,128+32+16+3])
        INPUTcoeff_np = tflib.run(INPUTcoeff, {latents: latent_np})
        INPUTcoeff_np[:,:160] = coef[:,:160]
        INPUTcoeff_list.append(INPUTcoeff_np)

    render_img_list = []
    for INPUT_coef in INPUTcoeff_list:
        render_img_np = tflib.run(render_img, {INPUTcoeff_pl: INPUT_coef})
        render_img_list.append(render_img_np)

    real_img = np.array(Image.open(img_file))        

    save_img_list = [real_img] + [render_img_np[0].astype(np.uint8) for render_img_np in render_img_list]
    save_file_name = os.path.join(save_img_dir, 'test_img_np_' + str(idx).zfill(3) + '.npy')
    print(save_file_name)
    np.save(save_file_name, save_img_list)

