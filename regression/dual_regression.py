import csv
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.applications.efficientnet import EfficientNetB3
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, Input, Lambda, MaxPooling2D, ReLU, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model

import ast

from efficientnetv2 import effnetv2_model

# %% Main function

path = "G:/M2_internship/dataset/pre-processing/test"

if __name__ == "__main__":
    avail_model = ['B0', 'B3', 'V2-B3', 'V2-S', 'InceptionV3', 'Xception', 'ResNet50V2']
    avail_size = [(512, 512), (512, 512), (300, 300), (224, 224), (512, 512), (299, 299), (224, 224)]

    
    chosen_model = avail_model[3]
    (new_w, new_h) = avail_size[avail_model.index(chosen_model)]
    
    # For the first stage, we need 30 (15 landmarks x 2)
    n_outputs = 30 
    
    # For the second stage, we need only 2
    n_outputs_correction = 2
    
    # V-2 need to be load_weights (status: developing...)
    # Other EfficientNet use load_model
    
    tf.keras.backend.clear_session()
    
    rows = []
    
    if chosen_model == 'V2-S':      
        model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=[224, 224, 3]),
            effnetv2_model.get_model('efficientnetv2-s', include_top=False, pretrained='efficientnetv2-s'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(n_outputs),
        ])
    
        h5_file_name = 'insect_wing_landmark_regression_EfficientNetV2-S_flipped_y.h5'
        model.load_weights(h5_file_name)
    
    elif chosen_model == 'V2-B3':      
        model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=[300, 300, 3]),
            effnetv2_model.get_model('efficientnetv2-b3', include_top=False, pretrained='efficientnetv2-b3'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(n_outputs),
        ])
    
        h5_file_name = 'insect_wing_landmark_regression_EfficientNet' + chosen_model + '_data_aug_flipped_y_250_epochs.h5'
        model.load_weights(h5_file_name)
        
    else:
        #h5_file_name = 'insect_wing_landmark_regression_Xception_flipped_y_250_epochs.h5'
        #h5_file_name = 'insect_wing_landmark_regression_InceptionV3_flipped_y_250_epochs.h5'
        #h5_file_name = 'insect_wing_landmark_regression_InceptionV3_512x512_flipped_y_250_epochs.h5'
        #h5_file_name = 'insect_wing_landmark_regression_ResNet50V2_flipped_y_250_epochs.h5'
        #h5_file_name = 'insect_wing_landmark_regression_EfficientNetB3_data_aug_flipped_y.h5' 
        #h5_file_name = 'insect_wing_landmark_regression_EfficientNet' + chosen_model + '_500_epochs.h5'  
        #h5_file_name = 'insect_wing_landmark_regression_EfficientNetB0_flipped_y.h5'
        
        #h5_file_name = 'insect_wing_landmark_regression_' + chosen_model + '_flipped_y_250_epochs' + '.h5'
        model = tf.keras.models.load_model(h5_file_name)
    
    # Better use /test only
    #file_name = random.choice(os.listdir(path))[:-4] + ".jpg"
    
# %% The first stage regression
    
    final_res = []
    
    for f in os.listdir(path):
        file_name = f[:-4] + ".jpg"
        
        #file_name = random.choice(os.listdir(path))[:-4] + ".jpg"
        
        img = cv2.imread(os.path.join(path, file_name))
        
        img_resized = cv2.resize(img, (new_w, new_h))
        
        # Convert the image array to a proper shape for the inference model
        img_resized = np.asarray([img_resized])
        
        # First stage regression (like region-proposal)
        y_pred_first_stage = model.predict(img_resized)       
       
        # The scaling parameters
        old_w = 1360
        old_h = 1024
        
        scale_x = old_w/new_w
        scale_y = old_h/new_h
    
        # A half side length of the bounding box
        k = 20
        
# %% The second stage regression
        
        # Second stage regression result
        second_res = []
        
        # Second stage regression (re-correction for the first stage result)
        #model_second = tf.keras.models.load_model('second_stage_regression.h5')
        model_second = tf.keras.models.load_model('second_stage_regression_AlexNet_ft.h5')
        
        for i in range(0, len(y_pred_first_stage[0]), 2):
            a = int(y_pred_first_stage[0][i] * scale_x)
            b = int(y_pred_first_stage[0][i+1] * scale_y)
                        
            x_min = a - k 
            y_min = b - k
            x_max = a + k
            y_max = b + k
            
            crop_img = img[y_min:y_max, x_min:x_max]
            
            if len(crop_img) == 0:
                continue
            else:
                y_pred_second_stage = model_second.predict(np.reshape(crop_img, (1, 40, 40, 3)))
            
                second_res.append(y_pred_second_stage)
        
        second_res_flatten = np.concatenate(np.concatenate(second_res)).reshape(1, 30)
        second_res_flatten = [x - 20 for x in second_res_flatten]
        
# %% Synthesis the result from dual regression
    
        final_res = []    
    
        for i in range(0, len(y_pred_first_stage[0]), 2):
            a = int(y_pred_first_stage[0][i] * scale_x) + second_res_flatten[0][i]
            b = int(y_pred_first_stage[0][i+1] * scale_y) + second_res_flatten[0][i+1]
            
            final_res.append(a)
            final_res.append(b)
            
        rows.append(final_res)

# %% Save to .csv
    
    with open('eval_dual_regression_effnetv2-s_AlexNet_ft.csv', mode='w', newline='') as f:
        write = csv.writer(f)
        write.writerows(rows)
    
# %% Ploting

    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    
    for i in range(0, int(len(final_res)/2)):
        draw.point((y_pred_first_stage[0][2*i] * scale_x, y_pred_first_stage[0][2*i+1] * scale_y), 'red')
        draw.point((final_res[2*i], final_res[2*i+1]), 'blue')   

    
    im.show()