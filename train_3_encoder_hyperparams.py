# Copyright 2022 Adobe. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
import numpy as np

# Constant
from Util.training_util import FACE_ID_LOSS_TYPE
from Util.network_util import MODULATION_ENCODING, CO_MODULATION_MODE
DATASET_TYPE = ['FFHQ', 'Synthetic']
MODULATION_SPACE = ['W', 'W+']
LPIPS_IMAGE_SIZE = 256

# Params
synface_img_dir = '/home/code-base/user_space/Dataset/Self_Constructed_GAN_Render_Image_Pairs'
extreme_pose_dir = '/home/code-base/user_space/Dataset/Extreme_Pose_GAN_Render_Image_Pairs'
ffhq_train_img_dir = '/home/code-base/user_space/Dataset/FFHQ_DiscoFaceGAN_PreProcessed/Processed_File/train'
generated_img_size = 256
channel_multiplier = 2
latent = 512
n_mlp = 8
use_separate_D = True

tsr_encode = MODULATION_ENCODING[0]
tsr_train = True
w_encode = MODULATION_ENCODING[0]
w_train = True
w_plus_encode = MODULATION_ENCODING[1]
w_plus_encoder_layer_num = 18
w_plus_sliced_layer = None
w_plus_train = True
co_mod = CO_MODULATION_MODE[0]

use_tanh = False
ckpt = './Experiment_To_Keep/Exp_2021-10-10_19:16:42/ckpt/140000.pt'
load_train_state = True

gpu_device_ids = [4,5]
primary_device = 'cuda:4'

training_iters = 420001
rec_dataset_type = DATASET_TYPE[0]
ds_dataset_type = DATASET_TYPE[1]
ds_freq = 2 # Do 1 dual/contrastive supervision every ds_freq iterations
ex_ds_freq = 3 # Do 1 extreme ds supervision every ex_ds_freq ds iterations
rec_batch = 16
ds_batch = 16
init_lr = 0.001

use_g_reg = True
g_reg_freq = 4
generator_path_reg_weight = 2
path_reg_batch_shrink = 2
discriminator_r1 = 10
d_reg_freq = 16

lpips_loss_lambda = 3
l1_loss_lambda = 3
ep_lpips_l1_weight_shrink = 10
face_id_loss_lambda = 30
face_id_loss_type = FACE_ID_LOSS_TYPE[0]
hmap_loss_lambda = 0
hmap_iter_thres = np.inf
rec_face_reg_loss_lambda = 0
ds_face_reg_loss_lambda = 20
ep_face_reg_loss_lambda = 100

model_save_freq = 10000
quant_eval_img_dir = '/home/code-base/user_space/Dataset/FFHQ_DiscoFaceGAN_PreProcessed/Processed_File/val'
quant_eval_batch_size = 64
visual_real_img_dir = '/home/code-base/user_space/Dataset/Self_Constructed_Visual_Val_Imgs_3DGAN'
n_real_eval_faces = 2
n_syn_eval_faces = 2
val_sample_freq = 1000

