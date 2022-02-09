import cv2
from PIL import Image
import itertools
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from skimage.feature import peak_local_max
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys

# %%

def display_debug(image):
    cv2.imshow("Debug", image)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    return 0

def vector_encoder(v, w, h):
    # Input: 1. [v   ]: A vector that contains (x,y) coordinates of landmarks
    #        2. [w, h]: Original image with size w x h
       
    if len(v) % 2 != 0:
        raise Exception('Input vector must has even length')
    
    # Initialize 
    mat = np.zeros((h, w, 3), dtype=np.uint8)   
    
    # Create a mask
    mask = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
    
    # Fill the mask
    # To preserve the class order, we can assign each class with a specific color
    # i.e. Class 0  - Brightest pixel
    #      Class 15 - Darkest pixel
    for i in range(0, len(v), 2):
        # NOTE: x, y ~ height, width
        mask[round(v[i+1]), round(v[i])] = 255 - i*5
    
    # Display for debugging purpose ONLY
    #display_debug(mask)
    
    return mask
    
def vector_decoder(mask):
    # 15 (x, y) pairs of coordinate
    NUM_CLASSES = 15

    # Take care of the min_distance as it is a threshold of anti-noise pixels
    # Stretched images need higher min_distance
    for threshold in range(5,10):    
        coordinates = peak_local_max(mask, min_distance=threshold)
        if len(coordinates) == NUM_CLASSES:
            break
    
    pixel_value = []

    # Swap y and x in coordinates
    coordinates[:, [1,0]] = coordinates[:, [0,1]]    
    
    for i in range(0,len(coordinates)):
        pixel_value.append(mask[coordinates[i][0]][coordinates[i][1]])
    
    sorted_coord = list(zip(*sorted(zip(coordinates, pixel_value), key = lambda x:x[1], reverse=True)))

    augmented_vector = list(np.asarray(sorted_coord[0]).flatten())
    
    return augmented_vector

# %% Main function

if __name__ == "__main__":
    print("DEBUGGING")
    # mask_file = Image.open('D:/USTH_Master/Insect-wing/test_aug/output/test_aug_original_test_mask.png_8567b9e9-2900-4b28-9510-d0065c7ea988.png').convert('L')
    mask_file = Image.open('D:/USTH_Master/Insect-wing/dataset/augmented/pipeline_output/_groundtruth_(1)_all_egfr_F_R_oly_2X_1.jpg_0ccd907b-3f19-4da7-bf9a-0ed604e628b5.jpg').convert('L')
    mask = np.asarray(mask_file)
    
    print(vector_decoder(mask))
    print(len(vector_decoder(mask)))