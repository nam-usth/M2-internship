import cv2
import csv
from PIL import Image
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import sys

import ast

# %% Configuration

path = "D:/USTH_Master/Insect-wing/dataset/original/"
path_train = path + "train/"
path_test = path + "test/"
dirs = os.listdir(path)

aug_path = "D:/" 

# %%

def read_dual_data():
    # Read both original and augmented data
    
    # 15 (x, y) pairs of coordinate
    NUM_CLASSES = 15
    
    # Original file
    original_csv_file = path + "original_data.csv"
    original_data = pd.read_csv(original_csv_file, header=None)
    
    folder_type = original_data.iloc[:,0].to_list()
    image_name = original_data.iloc[:,1].to_list()
    coord_vector = original_data.iloc[:,2].to_list()
    
    # Augmented file
    augmented_csv_file = aug_path + "augmented_data.csv"
    aug_data = pd.read_csv(augmented_csv_file, header=None)
    
    aug_folder_type = aug_data.iloc[:,0].to_list()
    aug_image_name = aug_data.iloc[:,1].to_list()
    aug_coord_vector = aug_data.iloc[:,2].to_list()

    # Set folder type for augmented images
    for k in range(0, len(image_name)):    
        for i in range(0, len(aug_image_name)):
            if image_name[k] in aug_image_name[i]:
                aug_folder_type[i] = folder_type[k]
            
    # Standardized coords' length
    for i in range(0, len(aug_image_name)):
        refined_coord = ast.literal_eval(aug_coord_vector[i])
        
        # Trim/Extended coord
        if len(refined_coord) > (NUM_CLASSES*2):
            refined_coord = refined_coord[0:(NUM_CLASSES*2)]
        elif len(refined_coord) < (NUM_CLASSES*2):
            refined_coord = refined_coord + [-1]*(NUM_CLASSES*2 - len(refined_coord))

        aug_coord_vector[i] = refined_coord
    
    return [folder_type, image_name, coord_vector, aug_folder_type, aug_image_name, aug_coord_vector]
    
def create_refined_csv(folder_type, image_name, coord_vector, aug_folder_type, aug_image_name, aug_coord_vector):
    
    # Write to .csv file
    path_output = path + "refined_output/"
    os.chdir(path_output)
    
    
    
    with open('refined_data.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        rows = zip(folder_type + aug_folder_type, image_name + aug_image_name, coord_vector + aug_coord_vector)
        for row in rows:
            writer.writerow(row)
        
# %% Main function

if __name__ == "__main__":
    a, b, c, d, e, f =  read_dual_data()
    new_b = [s + '.jpg' for s in b]
    create_refined_csv(a, new_b, c, d, e, f)