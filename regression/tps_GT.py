import os 
import numpy as np
import cv2

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
    
    # Flip y
    lst[:,1] = np.subtract(old_h, lst[:,1])
    
    lst[:,0] *= new_w/old_w
    lst[:,1] *= new_h/old_h
    
    lst = np.float32(lst).tolist()
    return lst

def data_preparation(img, coords, mode):
    
    # 2 inputs:
    # original image tensor
    # GT coords from .tps file

    old_w = 1360
    old_h = 1024

    # If mode = 1 means the user is running EfficientNetB0 and need to resize image + ground truth
    if (mode==1):
        new_w = 512 # 224
        new_h = 512 # 224
    else:
        new_w = 512
        new_h = 512
        
    altered_img = cv2.resize(img, (new_w, new_h))
    
    altered_GT = resize_GT(coords, old_w, old_h, new_w, new_h)
    return [altered_img, altered_GT]

# %%

def make_ready(mode):
    altered_data = []
    
    os.chdir('../dataset')
    
    for file in os.listdir('./original'):
        if(file[-3:]=="tps"):
            link = os.path.join("./original", file)
            tps1 = readtps(link)
            imagename = file[:-4]+".jpg"
            img = cv2.imread("./pre-processing/all/"+imagename)
            
            # get the coords vector with command: list(tps1.items())[2][1] or ...[2][1][0]
            # Be careful, we are doing wrong with the coord! [BUG]
            altered_data.append(data_preparation(img, list(tps1.items())[2][1][0], mode))
            
    os.chdir('../regression')
    
    return altered_data

