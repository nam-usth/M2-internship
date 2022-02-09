import random

import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = "D:/USTH_Master/Insect-wing/dataset/original/refined_output/"

csv_file = path + "refined_data.csv"
data = pd.read_csv(csv_file, header=None)

folder_type = data.iloc[:,0].to_list()
image_name = data.iloc[:,1].to_list()
coord_vector = data.iloc[:,2].to_list()

row_count = []

for i in range(1133, len(image_name)):
    if image_name[i][13:][:-41] != image_name[i-1][13:][:-41]:
        row_count.append(i)

selected_row = []

for j in range(0, len(row_count)-1):
    randomlist = random.sample(range(row_count[j], row_count[j+1]), 2)
    selected_row.append(randomlist)    
    
flat_sr = [item for sublist in selected_row for item in sublist]

flat_sr.sort()

df = pd.concat([data.iloc[0:1132, :], data.iloc[flat_sr,:]])

df.to_csv(path + 'refined_data_lite_3k.csv')