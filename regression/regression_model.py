import csv
import datetime
import itertools
import numpy as np
import os
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.applications.efficientnet import EfficientNetB3
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, Input, Lambda, MaxPooling2D, ReLU, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical

import tps_GT

# %% Define base model for transfer learning

def create_model(mode):
    # Number of output values: 
    #       30 values of x and y 
    #       ~ 15 pairs of [ [x1, y1], [x2, y2], [x3, y3], ..., [x15, y15] ]
    n_outputs = 30 
    
    name = 'insect_wing_landmark_regression'
    
    if (mode==0):
        name += '_customCNN'
        
        # A simple Regression custom Convolution Neural Network
        model = Sequential()
        
        model.add(Conv2D(64, (5, 5), padding='same', activation='relu', input_shape=(512, 512, 3)))
        model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
        #model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu')) 
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu')) 
        #model.add(Conv2D(128, (3, 3), padding='same', activation='relu')) 
        model.add(MaxPooling2D(pool_size=(2, 2))) 
        
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu')) 
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu')) 
        #model.add(Conv2D(256, (3, 3), padding='same', activation='relu')) 
        model.add(MaxPooling2D(pool_size=(2, 2))) 
        
        model.add(Flatten()) 
        model.add(Dense(n_outputs, activation='relu')) 
        model.add(Dropout(0.4)) 
    
    else:
        name += '_EfficientNetB0'
        
        # NOTE: Resize image to (224, 224, 3), also apply to the groundtruth (changing 15 pairs of coordinate 
        #       to fit the resized image)
        
        # Use Transfer Learning with EfficientNetB0
        
        # inputs = Input(shape=(224, 224, 3))
        inputs = Input(shape=(512, 512, 3))
        model = EfficientNetB0(input_tensor=inputs, weights='efficientnetb0_notop.h5', include_top=False)
        
        # Freeze the pretrained weights
        model.trainable = False
    
        # Rebuild top
        x = GlobalAveragePooling2D()(model.output)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        outputs = Dense(n_outputs)(x)
    
        # Compile
        model = Model(inputs, outputs, name="EfficientNet")
        
    return model, name
    
# %% Main function

if __name__ == "__main__":
       
    # Initialize a base_model
    # NOTE: mode = 0 --> Activate simple Regression custom CNN
    #       mode = 1 --> Activate adapt the EfficientNetB0 to a regression problem
    
    mode = 1
    
    # Getting data ready
    full_data = tps_GT.make_ready(mode)
    
    X, y = [], []
    
    for i in range(0, len(full_data)):
        X.append(full_data[i][0])
        y.append(full_data[i][1])
    
    for i in range(0, len(y)):
        if len(y[i]) > 15:
            y[i] = y[i][0:15]
            
    y_flatten = np.concatenate(np.concatenate(y)).reshape(1133, 30)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_flatten, test_size=0.2, random_state=42)
    
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    
    model, saved_name = create_model(mode)
        
    model.summary()
    
    tensorboard_callback = TensorBoard(log_dir="logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    opt = tf.keras.optimizers.SGD(learning_rate=1e-3) # Take care of the learning_rate
    model.compile(loss='mse', optimizer=opt, metrics=[tf.keras.metrics.MeanSquaredError()])
    
    history = model.fit(X_train, y_train, epochs=500, validation_split=0.1, callbacks=[tensorboard_callback], batch_size=16)
      
    # Saving trained model 
    h5_file_name = saved_name + '_flipped_y.h5'
    model.save(h5_file_name)
    
    # Load the saved model - use ONLY when inference (not to re-train again/call specific saved model)
    # model = tf.keras.models.load_model("insect_wing_landmark_regression_EfficientNetB0.h5")
    
    # Predict a cropped image/Calculate recall, precision and F1 score
    y_pred = model.predict(X_test)    
    
    # Write prediction result to .csv file
    os.chdir('../regression/result')
    pd.DataFrame(y_pred).to_csv("y_pred.csv")
