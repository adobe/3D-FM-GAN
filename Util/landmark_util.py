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

import sys
sys.path.append('/home/code-base/user_space/2021_work/third_party_code/face-alignment/')
from face_alignment.detection.sfd.detect import get_predictions
from face_alignment.utils import transform, get_preds_fromhm

def Batch_Img_Face_Detection(img_tensor, face_detector, device):
    '''
    Usage:
        Return N bounding boxes for N single-face images, each image should have only one detected faces  
    
    Args:
        img_tensor:    (torch.Tensor) of shape [N, 3, 256, 256] range from [0, 255]
        face_detector: (SFDDetector) defined in face_alignment.detection.sfd.sfd_detector
        
    '''

    img = img_tensor.flip(-3)
    img = img - torch.tensor([104.0, 117.0, 123.0], device=device).view(1, 3, 1, 1)
    
    # Batch inference
    olist = face_detector.face_detector(img)
    
    for i in range(len(olist) // 2):
        olist[i * 2] = F.softmax(olist[i * 2], dim=1)
    olist = [oelem.data.cpu().numpy() for oelem in olist]
    
    bbox_list = []
    for img in range(img.shape[0]):
        olist_img = [oelem[img:img+1,...] for oelem in olist]
        bboxlists = get_predictions(olist_img, 1)[0]
        bboxlists = face_detector._filter_bboxes(bboxlists)
        if len(bboxlists) == 0:
            bbox_list.append([0,0,255,255,1])
        elif (bboxlists[0][0] < 0) or (bboxlists[0][1] < 0) or (bboxlists[0][2] > 255) or (bboxlists[0][3] > 255):
            bbox_list.append([0,0,255,255,1])
        else:        
            bbox_list.append(bboxlists[0])
    return bbox_list


def Crop_PyTorch(img_tensor, center, scale, resolution = 256):
    '''
    Usage:
        Crop a pytorch tensor based on the face bounding box
    
    Args:
        img_tensor: (torch.Tensor) one image of shape [1, 3, 256, 256]
        center:     (float) the center of the faces
        scale:      (float) scale of the face
    '''
    ul = transform([1, 1], center, scale, resolution, True)
    br = transform([resolution, resolution], center, scale, resolution, True)
    
    _, c, ht, wd = img_tensor.shape
    
    newDim = ([1, c, br[1] - ul[1], br[0] - ul[0]])
    new_img_tensor = torch.zeros(newDim)
    
    # Define new dimension
    newX = np.array([max(1, -ul[0] + 1), min(br[0], wd) - ul[0]], dtype=np.int32)
    newY = np.array([max(1, -ul[1] + 1), min(br[1], ht) - ul[1]], dtype=np.int32)
    oldX = np.array([max(1, ul[0] + 1), min(br[0], wd)], dtype=np.int32)
    oldY = np.array([max(1, ul[1] + 1), min(br[1], ht)], dtype=np.int32)        

    new_img_tensor[..., newY[0]-1 : newY[1], newX[0]-1 : newX[1]
           ] = img_tensor[..., oldY[0]-1 : oldY[1], oldX[0]-1 : oldX[1]]
    
    new_img_tensor = F.interpolate(new_img_tensor, (resolution, resolution), mode = 'bilinear')
    return new_img_tensor


def Crop_An_Image(img_tensor, bbox, reference_scale):
    '''
    Usage:
        A wrapper for Pytorch Tensor cropping
        
    Args:
        img_tensor:      (torch.Tensor) one image of shape [1, 3, 256, 256]
        bbox:            (list) of 5 elements, first 4 define the bounding box
        reference_scale: (float) for the scale adjustment 
    '''
    
    center = torch.tensor([bbox[2] - (bbox[2] - bbox[0]) / 2.0, bbox[3] - (bbox[3] - bbox[1]) / 2.0])
    center[1] = center[1] - (bbox[3] - bbox[1]) * 0.12
    scale = (bbox[2] - bbox[0] + bbox[3] - bbox[1]) / reference_scale
    
    crop_tensor = Crop_PyTorch(img_tensor, center, scale)
    return crop_tensor, center, scale



def Get_Preds_FromHeatMap_PyTorch(hm, device, center_list=None, scale_list=None):
    """Obtain (x,y) coordinates given a set of N heatmaps. If the center_list
    and the scale is provided the function will return the points also in
    the original coordinate frame.
    Arguments:
        hm {torch.tensor} -- the predicted heatmaps, of shape [B, N, W, H]
    Keyword Arguments:
        center_list: (list) of (torch.tensor) -- the center of the bounding box for each image, len would be B
        scale_list:  (list) of (float) -- face scale, len would be B
    """
    B, C, H, W = hm.shape
    hm_reshape = hm.reshape(B, C, H * W)
    idx = (torch.argmax(hm_reshape, axis=-1)).float()
    
    preds, preds_orig = _get_preds_fromhm_torch(hm, idx, device, center_list, scale_list)

    return preds, preds_orig



def _get_preds_fromhm_torch(hm, idx, device, center_list=None, scale_list=None):
    """Obtain (x,y) coordinates given a set of N heatmaps and the
    coresponding locations of the maximums. If the center
    and the scale is provided the function will return the points also in
    the original coordinate frame.
    Arguments:
        hm {torch.tensor} -- the predicted heatmaps, of shape [B, N, W, H]
    Keyword Arguments:
        center_list: (list) of (torch.tensor) -- the center of the bounding box for each image, len would be B
        scale_list:  (list) of (float) -- face scale, len would be B
    """
    B, C, H, W = hm.shape
    idx += 1
    preds = idx.repeat_interleave(2).reshape(B, C, 2)
        
    preds[:, :, 0] = (preds[:, :, 0] - 1) % W + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / H) + 1
        
    for i in range(B):
        for j in range(C):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = torch.tensor(
                    [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                     hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                preds[i, j] += (torch.sign(diff) * 0.25).to(device)

    preds -= 0.5
    preds_orig = torch.zeros_like(preds)
    
    if center_list is not None and scale_list is not None:
        for i in range(B):
            center = center_list[i]
            scale = scale_list[i]
            for j in range(C):
                preds_orig[i, j] = transform(preds[i, j], center, scale, H, True)

    
    return preds, preds_orig


# ============================ Wrapper Function ============================

def Get_HeatMap_PyTorch(img_tensor, fa, device):
    '''
    Usage:
        Input a PyTorch image tensor and return its heatmaps
    
    Args:
        img_tensor: (torch.Tensor) of shape [N, 3, 256, 256] with the range of [-1, 1]
        fa:         (FaceAlignment) defined in face_alignment.FaceAlignment
        device:     (str) name of the device to place the tensor
        
    '''    

    # BBox Detection
    img_tensor = (img_tensor + 1) * 255 / 2 # Convert it back to the range of [0, 255]
    bbox_list = Batch_Img_Face_Detection(img_tensor, fa.face_detector, device) # one image one BB
    
    # Image Cropping
    crop_tensor_list = []
    center_list = []
    scale_list = []
    for i in range(img_tensor.shape[0]):
        slice_img_tensor = img_tensor[i:i+1]
        crop_tensor, center, scale = Crop_An_Image(slice_img_tensor, bbox_list[i], fa.face_detector.reference_scale)
        crop_tensor_list.append(crop_tensor)
        center_list.append(center)
        scale_list.append(scale)
    
    crop_tensor_batch = (torch.cat(crop_tensor_list) / 255.0).to(device)
    
    # HeatMap Prediction
    heatmap = fa.face_alignment_net(crop_tensor_batch)

    return heatmap, center_list, scale_list


def Get_HeatMap_Landmark_PyTorch(img_tensor, fa, device):
    '''
    Usage:
        Input a PyTorch image tensor and return its heatmaps and landmarks 
    
    Args:
        img_tensor: (torch.Tensor) of shape [N, 3, 256, 256] with the range of [-1, 1]
        fa:         (FaceAlignment) defined in face_alignment.FaceAlignment
        device:     (str) name of the device to place the tensor
    
    Returns:
        heatmap:    (torch.Tensor) of shape [N, 68, 64, 64]
        landmark:   (np.array) of shape [N, 68, 2]
    '''    
    # Get the HeatMap
    heatmap, center_list, scale_list = Get_HeatMap_PyTorch(img_tensor, fa, device)
    
    # Landmark from HeatMap
    heatmap_np = heatmap.detach().cpu().numpy()
    landmark = []
    for i in range(heatmap_np.shape[0]):
        hm = heatmap_np[i: i+1]
        center = center_list[i].numpy()
        scale = scale_list[i]
        pts, pts_img, scores = get_preds_fromhm(hm, center, scale)
        landmark.append(pts_img)
    
    landmark = np.concatenate(landmark)
    
    return heatmap, landmark


def Get_Landmark_PyTorch(img_tensor, fa, device):
    '''
    Usage:
        Input a PyTorch image tensor and return its heatmaps and landmarks 
    
    Args:
        img_tensor: (torch.Tensor) of shape [N, 3, 256, 256] with the range of [-1, 1]
        fa:         (FaceAlignment) defined in face_alignment.FaceAlignment
        device:     (str) name of the device to place the tensor
    
    Returns:
        landmark:   (np.array) of shape [N, 68, 2]
    '''    
    # Get the HeatMap
    heatmap, center_list, scale_list = Get_HeatMap_PyTorch(img_tensor, fa, device)
    
    # Landmark from HeatMap
    _, landmark = Get_Preds_FromHeatMap_PyTorch(heatmap, device, center_list, scale_list)
    
    return landmark
