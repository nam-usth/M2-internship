import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Model, load_model

from efficientnetv2 import effnetv2_model

# %% Main function

path = "G:/M2_internship/dataset/pre-processing/test"
#path = "D:/USTH_Master/Insect-wing/dataset/original/refined_output/image"

if __name__ == "__main__":
    avail_model = ['B0', 'B3', 'V2-B3', 'V2-S', 'InceptionV3', 'Xception', 'ResNet50V2']
    avail_size = [(224, 224), (512, 512), (300, 300), (224, 224), (512, 512), (299, 299), (224, 224)]

    
    chosen_model = avail_model[4]
    (new_w, new_h) = avail_size[avail_model.index(chosen_model)]
    
    n_outputs = 30 
    
    # V-2 need to be load_weights (status: developing...)
    # Other EfficientNet use load_model
    
    tf.keras.backend.clear_session()
    
    if chosen_model == 'V2-S':      
        model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=[224, 224, 3]),
            effnetv2_model.get_model('efficientnetv2-s', include_top=False, pretrained='efficientnetv2-s'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(n_outputs),
        ])
    
        h5_file_name = 'insect_wing_landmark_regression_EfficientNet' + chosen_model + '_250_epochs' + '.h5'
        model.load_weights(h5_file_name)
    
    elif chosen_model == 'V2-B3':      
        model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=[300, 300, 3]),
            effnetv2_model.get_model('efficientnetv2-b3', include_top=False, pretrained='efficientnetv2-b3'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(n_outputs),
        ])
    
        h5_file_name = 'insect_wing_landmark_regression_EfficientNet' + chosen_model + '_data_aug_flipped_y_250_epochs' + '.h5'
        model.load_weights(h5_file_name)
        
    else:
        #h5_file_name = 'insect_wing_landmark_regression_Xception_flipped_y_250_epochs.h5'
        #h5_file_name = 'insect_wing_landmark_regression_InceptionV3_flipped_y_250_epochs.h5'
        h5_file_name = 'insect_wing_landmark_regression_InceptionV3_512x512_flipped_y_250_epochs.h5'
        #h5_file_name = 'insect_wing_landmark_regression_ResNet50V2_flipped_y_250_epochs.h5'
        #h5_file_name = 'insect_wing_landmark_regression_EfficientNetB3_data_aug_flipped_y.h5' 
        #h5_file_name = 'insect_wing_landmark_regression_EfficientNet' + chosen_model + '_500_epochs' + '.h5'   
        model = tf.keras.models.load_model(h5_file_name)
    
    # Better use /test only
    file_name = random.choice(os.listdir(path))[:-4] + ".jpg"

    img = cv2.imread(os.path.join(path, file_name))
    
    img_resized = cv2.resize(img, (new_w, new_h))
    
    # Convert the image array to a proper shape for the inference model
    img_resized = np.asarray([img_resized])
    
    y_pred = model.predict(img_resized)
    
    # Draw predicted landmarks
    old_w = 1360
    old_h = 1024
    
    scale_x = old_w/new_w
    scale_y = old_h/new_h
    
    for i in range(0, int(len(y_pred[0])/2)):
        x = int(y_pred[0][2*i] * scale_x)
        #y = old_h - int(y_pred[0][2*i + 1] * scale_y) # Why does it flip? .TPS does flip y
        y = int(y_pred[0][2*i + 1] * scale_y)
        
        img = cv2.circle(img, (x,y), radius=5, color=(255, 0, 0), thickness=2)
    
    plot = plt.figure(1)
    plt.imshow(img)

    cv2.imshow('Landmark regression', img)
    
    cv2.waitKey(0)