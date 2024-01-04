# Copyright 2022 Adobe. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
import cv2
import os
from PIL import Image
import numpy as np
import time

from mtcnn import MTCNN

def Write_Detection_Result(detect_result, lm_path):
    '''
    Usage:
        Write the detection result from MTCNN to .txt format
    Args:
        detect_result: (dict) of the detection results
        lm_path: (str) of the file path with .txt as the extension
    '''
    
    mfile = open(lm_path, 'w')
    for val in detect_result[0]['keypoints'].values():
        str_to_write = str(val[0]) + ' ' + str(val[1]) + '\n'
        mfile.write(str_to_write)

FFHQ_Path = '''/home/code-base/user_space/Dataset/FFHQ_Amazon/'''
FFHQ_LM_Path = '''/home/code-base/user_space/Dataset/FFHQ_DiscoFaceGAN_PreProcessed/LandMark/'''
detector = MTCNN()

start_time = time.time()
for img_file in os.listdir(FFHQ_Path):
    try:
        img_path = os.path.join(FFHQ_Path, img_file) 
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        detect_result = detector.detect_faces(img)
        lm_path = os.path.join(FFHQ_LM_Path, img_file.replace('png', 'txt'))
        
        Write_Detection_Result(detect_result, lm_path)
    except:
        print('Oops, image file: ' + str(img_file) + ' has processing issues.')

end_time = time.time()
print('Total process time: ' + str(round(end_time - start_time, 2)))
