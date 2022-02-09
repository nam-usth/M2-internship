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

# %% Read the groundtruth info

path_GT = "D:/USTH_Master/Insect-wing/non_resized_dataset"
file_GT = os.path.join(path_GT, "original_data.csv")

df_GT = pd.read_csv(file_GT)

image_name_GT = df_GT.iloc[:,1].to_list()
lm_GT = df_GT.iloc[:,2].to_list()
y_test = []

for k in range(0,len(lm_GT)):
    y_test.append(ast.literal_eval(lm_GT[k]))
    
# %% Mini AlexNet with fine-tuning

def AlexNet_ft():
  model_input = Input(shape=(40, 40, 3))

  # First layer
  x = Conv2D(filters=72, kernel_size=(5, 5), strides=(1, 1), padding='same', name="conv1", activation="relu")(model_input)
  x = BatchNormalization()(x)
  x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="pool1")(x)

  # Second layer
  x = Conv2D(filters=96, kernel_size=(5, 5), strides=(1, 1), padding='same', name="conv2", activation="relu")(x)
  x = BatchNormalization()(x)
  x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="pool2")(x)

  x = Flatten()(x)

  x = Dense(384, activation='relu', name="fc3")(x)
  x = Dropout(0.2, name="droupout3")(x)
  x = Dense(192, activation='relu', name="fc4")(x)
  x = Dropout(0.2, name="droupout4")(x)
  x = Dense(2, name="fc5")(x)

  model = Model(inputs=model_input, outputs=x)

  return model

# %% Main function

path = "G:/M2_internship/dataset/pre-processing/all"

if __name__ == "__main__":
    avail_model = ['B0', 'B3', 'V2-B3', 'V2-S', 'InceptionV3', 'Xception', 'ResNet50V2']
    avail_size = [(224, 224), (512, 512), (300, 300), (224, 224), (512, 512), (299, 299), (224, 224)]

    
    chosen_model = avail_model[4]
    (new_w, new_h) = avail_size[avail_model.index(chosen_model)]
    
    # For the first stage, we need 30 (15 landmarks x 2)
    n_outputs = 30 
    
    # For the second stage, we need only 2
    n_outputs_correction = 2
    
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
    
        h5_file_name = 'insect_wing_landmark_regression_EfficientNet' + chosen_model + '_data_aug_flipped_y_250_epochs.h5'
        model.load_weights(h5_file_name)
        
    else:
        #h5_file_name = 'insect_wing_landmark_regression_Xception_flipped_y_250_epochs.h5'
        #h5_file_name = 'insect_wing_landmark_regression_InceptionV3_flipped_y_250_epochs.h5'
        h5_file_name = 'insect_wing_landmark_regression_InceptionV3_512x512_flipped_y_250_epochs.h5'
        #h5_file_name = 'insect_wing_landmark_regression_ResNet50V2_flipped_y_250_epochs.h5'
        #h5_file_name = 'insect_wing_landmark_regression_EfficientNetB3_data_aug_flipped_y.h5' 
        #h5_file_name = 'insect_wing_landmark_regression_EfficientNet' + chosen_model + '_500_epochs.h5'  
        
        #h5_file_name = 'insect_wing_landmark_regression_' + chosen_model + '_flipped_y_250_epochs' + '.h5'
        model = tf.keras.models.load_model(h5_file_name)
    
    # Better use /test only
    #file_name = random.choice(os.listdir(path))[:-4] + ".jpg"
    
# %% Prepare the GT from the first stage

    image_write_path = "D:/USTH_Master/Insect-wing/cropped"
    image_write_file_name = []
    image_write_file_GT = []
        
    for f in os.listdir(path):
        file_name = f[:-4] + ".jpg"
        
        img = cv2.imread(os.path.join(path, file_name))
        
        img_resized = cv2.resize(img, (new_w, new_h))
        
        # Convert the image array to a proper shape for the inference model
        img_resized = np.asarray([img_resized])
        
        # First stage regression (like region-proposal)
        y_pred_first_stage = model.predict(img_resized)
        
        # Prepare the GT
        current_file_GT = y_test[image_name_GT.index(file_name)]
        
        # The scaling parameters
        old_w = 1360
        old_h = 1024
        
        scale_x = old_w/new_w
        scale_y = old_h/new_h
    
        # A half side length of the bounding box
        k = 20
        
        # Second stage regression (re-correction for the first stage result)
        for i in range(0, len(y_pred_first_stage[0]), 2):
            a = int(y_pred_first_stage[0][i] * scale_x)
            b = int(y_pred_first_stage[0][i+1] * scale_y)
            
            x_origin_GT = current_file_GT[i]
            y_origin_GT = current_file_GT[i+1]
            
            x_cropped_GT = x_origin_GT - a + k
            y_cropped_GT = y_origin_GT - b + k
            
            x_min = a - k if (a - k) > 0 else 0
            y_min = b - k if (b - k) > 0 else 0
            x_max = a + k if (a + k) < old_w else old_w
            y_max = b + k if (b + k) < old_h else old_h
            
            crop_img = img[y_min:y_max, x_min:x_max]
            
            s = os.path.join(image_write_path, file_name[:-4] + "_cropped_LM_" + str(i//2) + ".jpg")
            image_write_file_name.append(s)
            
            if x_cropped_GT < 0 or x_cropped_GT > 2*k:
                x_cropped_GT = 0
            if y_cropped_GT < 0 or y_cropped_GT > 2*k:
                y_cropped_GT = 0
            
            coords = [x_cropped_GT, y_cropped_GT]
            image_write_file_GT.append(coords)
            
            # Only uncomment this to write cropped images to disk
            # cv2.imwrite(s, crop_img)
            
        rows = zip(image_write_file_name, image_write_file_GT)

    # Only uncomment these line when we need to create a new .csv file
    '''        
    with open('second_regression_stage_train_annotate.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
    '''

    '''        
    # Draw predicted landmarks    
    
    for i in range(0, int(len(y_pred_first_stage[0])/2)):
        x = int(y_pred_first_stage[0][2*i] * scale_x)
        #y = old_h - int(y_pred_first_stage[0][2*i + 1] * scale_y) # Why does it flip? .TPS does flip y
        y = int(y_pred_first_stage[0][2*i + 1] * scale_y)
        
        img = cv2.circle(img, (x,y), radius=5, color=(255, 0, 0), thickness=2)
    
    plot = plt.figure(1)
    plt.imshow(img)

    cv2.imshow('Landmark regression', img)
    '''
    
    cv2.waitKey(0)
    
# %% Train the second stage

    df = pd.read_csv('D:/USTH_Master/Insect-wing/valid_image_with_annotate.csv')
    
    X, y = [], []
    
    image_file = df.iloc[:,0].to_list()
    groundtruth = df.iloc[:,1].to_list()
    
    for f in range(0,len(image_file)):
        X.append(np.asarray(Image.open(image_file[f])))
    
    for k in range(0,len(groundtruth)):
        y.append(ast.literal_eval(groundtruth[k]))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    '''
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(40, 40, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten()) 
    model.add(Dense(2))

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=opt, metrics=[tf.keras.metrics.MeanSquaredError()])
    '''
    
    model = Sequential()
    model = AlexNet_ft()
    model.summary()
    
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(loss='mse', optimizer=opt, metrics=[tf.keras.metrics.MeanSquaredError()])

    history = model.fit(X_train, y_train, epochs=50, validation_split=0.1, batch_size=64)
    
    # Saving trained model 
    h5_file_name = 'second_stage_regression_AlexNet_ft.h5'
    model.save(h5_file_name)
    
# %% Predict and draw

    y_pred = model.predict(X_test)
    
    index = 15
    
    im = Image.fromarray(X_test[index])
    draw = ImageDraw.Draw(im)  
    draw.point((y_pred[index][0], y_pred[index][1]), 'red')   
    im.show()