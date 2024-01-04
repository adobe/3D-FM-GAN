# Copyright 2022 Adobe. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
'''
This script will use DiscoFaceGAN to generate images with the same identity but multiple non-identity variations,
and iterate the process for multiple identities.
'''

import tensorflow as tf
import numpy as np
from PIL import Image
import os

# DiscoFaceGAN utils
import sys
sys.path.append('/home/code-base/user_space/2021_work/DiscoFaceGAN/')
import dnnlib.tflib as tflib
from generate_images import load_Gs, z_to_lambda_mapping, truncate_generation, restore_weights_and_initialize
from renderer.face_decoder import Face3D

# Constant 
LATENT_DIMENSION = 128+32+16+3
NOISE_DIMENSION = 32
LAMBDA_DIMENSION = 254
IDENTITY_SLICE = [0, 160]

# Specific Generation Params
saving_directory = '''/home/code-base/user_space/Dataset/Self_Constructed_GAN_Render_Image_Pairs'''
num_identity = 10000 
num_per_identity_variation = 7
rand_seed = 100

# Model Loading
tflib.init_tf()
with tf.device('/gpu:0'):
    url_pretrained_model_ffhq = 'https://drive.google.com/uc?id=1nT_cf610q5mxD_jACvV43w4SYBxsPUBq'
    Gs = load_Gs(url_pretrained_model_ffhq)
    average_w_id = None
    
    FaceRender = Face3D()

# Define the latent to lambda space mapping
with tf.device('/gpu:0'):
    latents = tf.placeholder(tf.float32, name='latents', shape=[1,LATENT_DIMENSION])
    noise = tf.placeholder(tf.float32, name='noise', shape=[1,NOISE_DIMENSION])
    INPUTcoeff = z_to_lambda_mapping(latents)
restore_weights_and_initialize()    

# Define the (GAN,Render) image pairs generation 
with tf.device('/gpu:0'):
    INPUTcoeff_pl = tf.placeholder(tf.float32, name='input_coeff', shape=[1,LAMBDA_DIMENSION])
    INPUTcoeff_w_noise = tf.concat([INPUTcoeff_pl,noise],axis = 1)
    fake_images_out = truncate_generation(Gs,INPUTcoeff_w_noise,dlatent_average_id=average_w_id)    
    
    # Renderer input and generated images
    INPUTcoeff_w_t = tf.concat([INPUTcoeff_pl, tf.zeros([1,3])], axis = 1)
    render_img,render_mask,render_landmark,_ = FaceRender.Reconstruction_Block(INPUTcoeff_w_t,256,1,progressive=False)  


# Image generation process
np.random.seed(rand_seed)

for identity in range(num_identity):
    if identity % 100 == 0:
        print('Generating Identity: ' + str(identity))
    id_save_dir = os.path.join(saving_directory, 'id_' + str(identity + 1).zfill(5))
    os.mkdir(id_save_dir)
    
    identity_latent = np.random.normal(size=[1, LATENT_DIMENSION])
    identity_noise = np.random.normal(size=[1, NOISE_DIMENSION])
    identity_lambda = tflib.run(INPUTcoeff, {latents: identity_latent})
    fake_img, render_img_np = tflib.run([fake_images_out, render_img], {INPUTcoeff_pl: identity_lambda, noise: identity_noise})
    
    fake_img_file = os.path.join(id_save_dir, 'g_1.png')
    render_img_file = os.path.join(id_save_dir, 'r_1.png')
    Image.fromarray(fake_img[0].astype(np.uint8)).save(fake_img_file, 'png')
    Image.fromarray(render_img_np[0].astype(np.uint8)).save(render_img_file, 'png')
    
    for variation in range(1, num_per_identity_variation):
        v_latent = np.random.normal(size=[1, LATENT_DIMENSION])
        v_lambda = tflib.run(INPUTcoeff, {latents: v_latent})
        
        v_lambda[0, IDENTITY_SLICE[0]:IDENTITY_SLICE[1]] = identity_lambda[0, IDENTITY_SLICE[0]:IDENTITY_SLICE[1]]
        fake_img, render_img_np = tflib.run([fake_images_out, render_img], {INPUTcoeff_pl: v_lambda, noise: identity_noise})
        
        fake_img_file = os.path.join(id_save_dir, 'g_' + str(variation + 1) + '.png')
        render_img_file = os.path.join(id_save_dir, 'r_' + str(variation + 1) + '.png')
        Image.fromarray(fake_img[0].astype(np.uint8)).save(fake_img_file, 'png')
        Image.fromarray(render_img_np[0].astype(np.uint8)).save(render_img_file, 'png')
