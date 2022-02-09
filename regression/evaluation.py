import csv
import os 
import numpy as np
import math 

def get_scale(mode):
    old_w = 1360
    old_h = 1024

    # If mode = 1 means the user is running EfficientNetB0 and need to resize image + ground truth
    if (mode==1):
        new_w = 512 # 224
        new_h = 512 # 224
    else:
        new_w = 512
        new_h = 512
    return [old_w/new_w, old_h/new_h]

# %% Main function

if __name__ == "__main__":
    mode = 1
    
    scale_x, scale_y = get_scale(mode)
    
    # Load data from .csv files:
    #y_test = []
    #y_pred = []
    
    res = []
    
    for i in range(0, len(y_pred)):
        temp = []
        
        d = [elem_pred - elem_test for (elem_pred, elem_test) in zip(y_pred[i], y_test[i])]
        
        it = iter(d)
        for diff_x, diff_y in zip(it, it):
            temp.append(math.sqrt((diff_x*scale_x)**2 + (diff_y*scale_y)**2))
            
        res.append(temp)
    
    # Save result to .csv file
    with open('eval.csv', mode='w', newline='') as f:
        write = csv.writer(f)
        write.writerows(res)