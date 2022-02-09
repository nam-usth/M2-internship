from PIL import Image
import os
import numpy as np
import pandas as pd
import csv
import ast

import data_enc_dec

import random
import Augmentor

# %% Configuration

path = "D:/USTH_Master/Insect-wing/dataset/original/"
path_train = path + "train/"
path_test = path + "test/"
dirs = os.listdir(path)

aug_path = "D:/USTH_Master/Insect-wing/dataset/augmented/"
aug_path_img = aug_path + "image/"
aug_path_mask = aug_path + "mask/"
aug_path_mask_train = aug_path_mask + "train/"
aug_path_mask_test = aug_path_mask + "test/"

# %%

def resize_image():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((512,512), Image.ANTIALIAS)
            imResize.save(f + '.jpg', 'JPEG', quality=90)
            
# %%

def readtps(input): 
    tps_file = open(input, 'r')  
    tps = tps_file.read().splitlines()  
    tps_file.close()

    lm, im, coords_array = [], [], []

    for i, ln in enumerate(tps):
        if ln.startswith("LM"):
            lm_num = int(ln.split('=')[1])
            lm.append(lm_num)
            coords_mat = []
            
            for j in range(i + 1, i + 1 + lm_num):
                coords_mat.append(tps[j].split(' ')) 

            coords_mat = np.array(coords_mat, dtype=float)
            coords_array.append(coords_mat)
           
        if ln.startswith("IMAGE"):
            im.append(ln.split('=')[1]) 

    return {'lm': lm, 'im': im, 'coords': coords_array} # return a dictionary

def resize_GT(lst, old_w, old_h, new_w, new_h):    
    lst = np.array(lst)
    
    lst[:,0] *= new_w/old_w
    lst[:,1] *= new_h/old_h
    
    lst = np.floor(lst).tolist()
    return lst

def get_vector(filename, flip_y="true"):
    old_w = 1360
    old_h = 1024
    
    new_w = 512
    new_h = 512
    
    os.chdir('G:/InsectWing')
        
    if (filename[-3:]=="tps"):
        link = os.path.join("./RightWingsFull", filename)
        tps1 = readtps(link)
    
        coords = list(tps1.items())[2][1][0]
     
    if flip_y:
        flipped_factor = [0, -new_h] * 15
    else:
        flipped_factor = [0, 0] * 15
        
    altered_GT = resize_GT(coords, old_w, old_h, new_w, new_h)
    
    coords_flatten = list(np.asarray(altered_GT).flatten())
    result = [abs(x + y) for x, y in zip(coords_flatten, flipped_factor)]
    
    return result

def create_original_csv():
    os.chdir(path)
    with open('original_data.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
    
        for item in os.listdir(path_train):
            if os.path.isfile(path_train+item):      
                f, e = os.path.splitext(path_train+item)
                
                row = ['TRAINING', item[:-4]+".jpg", get_vector(item[:-4]+".tps")]

                # write the data
                writer.writerow(row)
                
        for item in os.listdir(path_test):
            if os.path.isfile(path_test+item):      
                f, e = os.path.splitext(path_test+item)
                
                row = ['TEST', item[:-4]+".jpg", get_vector(item[:-4]+".tps")]

                # write the data
                writer.writerow(row)

# %%

def create_original_mask():
    original_csv_file = path + "original_data.csv"
    
    os.chdir(aug_path)
    
    data = pd.read_csv(original_csv_file, header=None)
    
    folder_type = data.iloc[:,0].to_list()
    image_name = data.iloc[:,1].to_list()
    coord_vector = data.iloc[:,2].to_list()
    
    for i in range(0, len(image_name)):        
        original_mask = data_enc_dec.vector_encoder(ast.literal_eval(coord_vector[i]), 512, 512)
        im = Image.fromarray(original_mask)
        
        if (folder_type[i] == 'TRAINING'):
            os.chdir(aug_path_mask_train)
        else:
            os.chdir(aug_path_mask_test)
            
        im.save(image_name[i]+".jpg")
        
def create_augmented_csv():   
    aug_path_output = aug_path + "pipeline_output/"
    os.chdir(aug_path)    
    with open('augmented_data.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        
        for item in os.listdir(aug_path_output):
            # for all mask file starts with "_groundtruth_(1)_all_" ~ 21 characters
            if item[:21] == '_groundtruth_(1)_all_':
                aug_image_name = "all_original_" + item[21:]

                aug_mask = np.asarray(Image.open(aug_path_output + item).convert('L'))
    
                row = ['', aug_image_name, data_enc_dec.vector_decoder(aug_mask)]
            
                # write the data
                writer.writerow(row)

# %%

def data_augmentation():
    p = Augmentor.Pipeline("D:/USTH_Master/Insect-wing/dataset/augmented/image/all/")
    p.ground_truth("D:/USTH_Master/Insect-wing/dataset/augmented/mask/all/")
    
    p.rotate(1, max_left_rotation=5, max_right_rotation=5)
    p.zoom_random(probability=0.5, percentage_area=0.8)
    p.flip_top_bottom(probability=0.5)

    p.sample(56650)

# %% Main function

if __name__ == "__main__":
    # Resize to (512, 512) for EfficientNet. Could be reduce to (299, 299) for InceptionV3 or (224, 224) for EfficientNetB3
    # resize_image()
    # create_original_csv()
    # create_original_mask()
    # data_augmentation()
    create_augmented_csv()
    