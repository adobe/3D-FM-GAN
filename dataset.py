# Copyright 2022 Adobe. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
import os

from PIL import Image

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler, Sized, Optional, Iterator

class FFHQ_Dataset(Dataset):
    '''
    Usage:
        Self-coded class for loading the FFHQ data
    '''
    
    def __init__(self, image_folder, transform = None):
        images_list = os.listdir(image_folder)
        self.images_list = sorted([os.path.join(image_folder, image) for image in images_list])
        self.transform = transform
    
    def __getitem__(self, index):
        img_id = self.images_list[index]
        img = Image.open(img_id).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        return img

    def __len__(self):
        return len(self.images_list)


class Synthetic_Dataset(Dataset):
    '''
    Usage:
        Self-coded class for loading pairs of (g_img, r_img) synthetic data
    '''
    
    def __init__(self, image_folder, transform = None):
        images_list = os.listdir(image_folder)
        self.id_list = sorted(os.listdir(image_folder))
        self.g_img_list = []
        self.r_img_list = []
        for pid in self.id_list:
            id_dir = os.path.join(image_folder, pid)
            self.g_img_list += [os.path.join(id_dir, image) for image in sorted(os.listdir(id_dir)) if 'g_' in image]
            self.r_img_list += [os.path.join(id_dir, image) for image in sorted(os.listdir(id_dir)) if 'r_' in image]
            
        self.transform = transform
    
    def __getitem__(self, index):
        g_img_path = self.g_img_list[index]
        r_img_path = self.r_img_list[index]
        
        g_img = Image.open(g_img_path).convert('RGB')
        r_img = Image.open(r_img_path).convert('RGB')
        
        if self.transform is not None:
            g_img = self.transform(g_img)
            r_img = self.transform(r_img)
        
        return g_img, r_img

    def __len__(self):
        return len(self.g_img_list)

class FFHQ_Dataset_Reconstruction(Dataset):
    '''
    Usage:
        Self-coded class for loading the FFHQ data
    '''
    
    def __init__(self, photo_image_folder, render_image_folder, transform = None):
        photo_image_list = os.listdir(photo_image_folder)
        render_image_list = os.listdir(render_image_folder)
        
        assert len(photo_image_list) == len(render_image_list)
        
        self.photo_image_list = [os.path.join(photo_image_folder, image) for image in sorted(photo_image_list)]
        self.render_image_list = [os.path.join(render_image_folder, image) for image in sorted(render_image_list)]
        self.transform = transform
    
    def __getitem__(self, index):
        photo_img_file = self.photo_image_list[index]
        render_img_file = self.render_image_list[index]
        
        photo_img = Image.open(photo_img_file).convert('RGB')
        render_img = Image.open(render_img_file).convert('RGB')
        
        if self.transform is not None:
            photo_img = self.transform(photo_img)
            render_img = self.transform(render_img)
        
        return photo_img, render_img

    def __len__(self):
        return len(self.photo_image_list)


class FFHQ_Dataset_Editing(Dataset):
    '''
    Usage:
        Self-coded class for loading the FFHQ data for quantitative image editing evaluation 
        and self-supervised contrastive supervision training
    '''
    
    def __init__(self, photo_image_folder, edit_render_image_folder, transform = None, train = False, render_image_folder = None):
        N_EDIT_IMG_PER_ID = 4
        
        photo_image_list = sorted(os.listdir(photo_image_folder))
        edit_render_image_list = sorted(os.listdir(edit_render_image_folder))
        
        assert (len(photo_image_list) * N_EDIT_IMG_PER_ID) == len(edit_render_image_list)
                
        self.photo_image_list = [os.path.join(photo_image_folder, image) for image in photo_image_list]
        self.edit_render_image_list = [os.path.join(edit_render_image_folder, image) for image in edit_render_image_list]
        
        # Group the edit render image in a better structured manner
        self.edit_render_image_list = [self.edit_render_image_list[N_EDIT_IMG_PER_ID * i: N_EDIT_IMG_PER_ID * (i + 1)] for i in range(len(self.photo_image_list))] 
        
        if train:
            render_image_list = sorted(os.listdir(render_image_folder))
            assert len(render_image_list) == len(photo_image_list)
            self.render_image_list = [os.path.join(render_image_folder, image) for image in render_image_list]
        
        self.transform = transform
        self.train = train
    
    def __getitem__(self, index):
        photo_img_file = self.photo_image_list[index]
        edit_render_img_file_list = self.edit_render_image_list[index]
        
        photo_img = Image.open(photo_img_file).convert('RGB')
        
        # Whether this dataset is built for training or evaluations
        if self.train: 
            edit_render_img_file = np.random.choice(edit_render_img_file_list) # Only select 1 edit_render_img for training
            render_img_list = [Image.open(file).convert('RGB') for file 
                               in [self.render_image_list[index], edit_render_img_file]]
            
        else:
            render_img_list = [Image.open(render_img_file).convert('RGB') for render_img_file in edit_render_img_file_list]
        
        if self.transform is not None:
            photo_img = self.transform(photo_img)
            render_img_list = [self.transform(render_img) for render_img in render_img_list]
        
        return [photo_img] + render_img_list

    def __len__(self):
        return len(self.photo_image_list)


# ----------------------- Dual Supervision Related Method -----------------------


def dual_supervision_list_augmentation(index_list, n_img_per_id):
    '''
    Usage:
        A function specific design for sampling of Synthetic_Dataset 
    
    Args:
        index_list:      (list) of a random permutation from [0, i - 1] of len i
        n_img_per_id:    (int) number of images per identity
    
    Returns:
        dual_index_list: (list) of a permutation from [0, 2i - 1] of len 2i 
                              where (2j, 2j + 1) are index pairs for dual supervison
    '''
    num_images = len(index_list) 
    dual_index_list = []

    for idx in index_list:
        person_id = idx // n_img_per_id
        non_identity_var_id = idx % n_img_per_id
        
        dual_non_identity_var_id = np.random.choice([i for i in range(n_img_per_id) if i != non_identity_var_id])
        dual_idx = person_id * n_img_per_id + dual_non_identity_var_id
        
        dual_index_list += [idx, dual_idx]

    return dual_index_list


class DualSupervisionSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
        generator (Generator): Generator used in sampling.
    """
    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, n_img_per_id: int, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None) -> None:
        self.data_source = data_source
        self.n_img_per_id = n_img_per_id
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got "
                            "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return 2 * len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        else:
            generator = self.generator

        index_list = torch.randperm(n, generator=generator).tolist()
        dual_index_list = dual_supervision_list_augmentation(index_list, self.n_img_per_id)
        yield from dual_index_list

    def __len__(self) -> int:
        return self.num_samples



# ----------------------- Extreme Pose Learning Related Method -----------------------

class ExtremePoseDualSupervisionSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.
    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
        generator (Generator): Generator used in sampling.
    """
    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, n_img_per_id: int, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None) -> None:
        self.data_source = data_source
        self.n_img_per_id = n_img_per_id
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got "
                            "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return 2 * len(self.data_source) // self.n_img_per_id
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source) // self.n_img_per_id
        if self.generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        else:
            generator = self.generator

        index_list = torch.randperm(n, generator=generator).tolist()
        extreme_pose_index_list = extreme_pose_list_augmentation(index_list, self.n_img_per_id)
        yield from extreme_pose_index_list

    def __len__(self) -> int:
        return self.num_samples


def extreme_pose_list_augmentation(index_list, n_img_per_id):
    '''
    Usage:
        A function designed for ExtremePoseDualSupervisionSampler, 
        to transform an identity index list to the actual image index list
        
    Args:
        index_list:       (list) of a random permutation of [0, i-1] where i is the number of identity
        n_img_per_id:     (int) number of images per identity
    
    Returns:
        extreme_idx_list: (list) of an extreme learning procedure where  
                          extreme_idx_list[j] is the image of normal pose if j is even number
                          extreme_idx_list[j] is the image of extreme pose if j is an odd number (0 indexing)
    '''
    
    extreme_idx_list = []
    
    for index in index_list:

        normal_pose_index = index * n_img_per_id
        extreme_idx_list.append(normal_pose_index)
        
        rand_offset = np.random.choice(np.arange(1, n_img_per_id))
        extreme_pose_index = normal_pose_index + rand_offset
        extreme_idx_list.append(extreme_pose_index)
    
    return extreme_idx_list



# ----------------------- Data Loading Pipeline -----------------------

def Swap_List_Pair(idx_list):
    '''
    Usage:
        Swap next to item of pair, like [0, 1, ..., n] -> [1, 0, ..., n, n-1], 
        where n is an even number
    
    Args:
        idx_list: (list) of [0, 1, ..., n]
    '''
    swap_list = []
    for idx, item in enumerate(idx_list):
        if idx % 2 == 0:
            swap_list.append(idx_list[idx + 1])
        else:
            swap_list.append(idx_list[idx - 1])
    return swap_list


def Data_Loading(rec_loader, ds_loader, ds_flag, device, extreme_loader = None, extreme_ds_flag = False,
                 pure_ffhq_loader = None, ds_dataset_type = None):
    '''
    Usage:
        Provide the data for learning in each round
    Args:
        rec_loader:     (DataLoader) to load data for reconstruction
        ds_loader:      (DataLoader) to load data for dual/constrastive supervision
        ds_flag:        (bool) whether this iteration is doing reconstruction or constrastive supervision
        device:         (str) the device to place the tensor on
        extreme_loader: (DataLoader) to load data with extreme pose
    '''

    if ds_dataset_type is None:
        if ds_flag is False: # reconstruction loading
            g_input, r_input = next(rec_loader)
            g_ref_np = g_input.detach().cpu().numpy()
            g_ref = torch.tensor(g_ref_np)

        elif (ds_flag is True) and (extreme_ds_flag is False): # normal disentangled editing
            g_input, r_input = next(ds_loader)

            # Swap the pair of images
            num_img = g_input.shape[0]
            swap_list = Swap_List_Pair(range(num_img))

            r_input = r_input[swap_list, ...]
            g_ref_np = g_input.detach().cpu().numpy()[swap_list, ...]
            g_ref = torch.tensor(g_ref_np)

        elif (ds_flag is True) and (extreme_ds_flag is True):
            g_input, r_input = next(extreme_loader)

            # Swap the pair of images
            num_img = g_input.shape[0]
            swap_list = Swap_List_Pair(range(num_img))

            r_input = r_input[swap_list, ...]
            g_ref_np = g_input.detach().cpu().numpy()[swap_list, ...]
            g_ref = torch.tensor(g_ref_np)
            
            # We only need even idices
            idx_slicing = np.arange(num_img // 2) * 2
            g_input, r_input, g_ref = g_input[idx_slicing, ...], r_input[idx_slicing, ...], g_ref[idx_slicing]
            
        return g_input.to(device), r_input.to(device), g_ref.to(device)

    elif ds_dataset_type == 'FFHQ':
        ffhq_ref = next(pure_ffhq_loader)
        g_input, r_input, r_edit_input = next(ds_loader)
        g_ref_np = g_input.detach().cpu().numpy()
        g_ref = torch.tensor(g_ref_np)
        return g_input.to(device), r_input.to(device), r_edit_input.to(device), g_ref.to(device), ffhq_ref.to(device)
