import cv2
import numpy as np
import os
import glob

import ast 
import math
import pandas as pd

classesFile = "landmark.names";
modelConfiguration = "yolov3-tiny-landmark.cfg";
modelWeights = "yolov3-tiny-landmark_20000.weights";
imgFolder = "G:/M2_internship/object_detection/images/test"

# Load Yolo
net = cv2.dnn.readNet(modelWeights, modelConfiguration)
classes = []
with open(classesFile, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
#output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# %%

row_pred = []

for filename in os.listdir(imgFolder):
    img = cv2.imread(os.path.join(imgFolder,filename))
    if img is None:
        continue
   
    #img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    #cv2.namedWindow('image',WINDOW_NORMAL)
    
    info = []
    
    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
                info.append([center_x, center_y, class_id, confidence])
                #print("Conf: ",confidence, " id:",class_id,"\n")
    
    info.sort(key = lambda x: x[2])
    #[k.pop(2) for k in info]
    row_pred.append([filename, info])

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)    
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 2, (0,0,0), 3)

    #cv2.imshow("Image", img)
    #ch = cv2.waitKey(0)    
    #if ch == 27 or ch == ord('q') or ch == ord('Q'):
    #    break

'''
df = pd.DataFrame(row_pred)
df.to_csv('./yolo_stat_pred.csv')    
'''

cv2.destroyAllWindows()

# %% Prepare the Groundtruth .csv

row_GT = []

for file in os.listdir("G:/M2_internship/object_detection/images/test"):
    if(file[-3:]=="tps"):
        continue
    if(file[-3:]=="jpg"):
        tps_file = open(os.path.join("G:/M2_internship/dataset/original", file[:-3]+"tps"), 'r')    
        tps = tps_file.read().splitlines()  
        tps_file.close()
        
        info = []
        lm, im, coords_array = [], [], []
        
        for i, ln in enumerate(tps):
            if ln.startswith("LM"):
                lm_num = int(ln.split('=')[1])
                lm.append(lm_num)
                coords_mat = []
                
                for j in range(i + 1, i + 1 + lm_num):
                    coords_mat.append(tps[j].split(' ')) 
    
                coords_mat = np.array(coords_mat, dtype=float)
                coords_mat = np.absolute(np.subtract(coords_mat, [0, 1024]))
                coords_array.append(coords_mat)
                        
        info.append(coords_array[0].tolist())
        row_GT.append([file[:-3]+"jpg", info])
        
df = pd.DataFrame(row_GT)
df.to_csv('./yolo_stat_GT.csv')  


# %% Benchmark

thres = 9

true_det_count = [0] * 15
false_det_count = [0] * 15

df_gt = pd.read_csv ('yolo_stat_GT.csv')
df_pred = pd.read_csv ('yolo_stat_pred.csv')

for i in range(0, len(row_pred)):
    flag = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    IMAGE_NAME = row_pred[i][0]
    for j in range(0, len(row_pred[i][1])):
        x_pred = row_pred[i][1][j][0]
        y_pred = row_pred[i][1][j][1]
        CLASS = row_pred[i][1][j][2]
        
        DF_COL_NAME = 'lm' + str(CLASS+1) + '_true'
        DF_ROW_INDEX = df_gt.index[df_gt['image_name']==IMAGE_NAME][0]
        GT_VAL = ast.literal_eval(df_gt.loc[DF_ROW_INDEX, DF_COL_NAME])
        
        x_true = GT_VAL[0]
        y_true = GT_VAL[1]
        
        dis = math.sqrt((x_true-x_pred)*(x_true-x_pred)+(y_true-y_pred)*(y_true-y_pred))       

        if flag[CLASS]==0:  
            if dis <= thres:
                true_det_count[CLASS] += 1
                flag[CLASS]=1
            else:
                false_det_count[CLASS] += 1
                flag[CLASS]=2
            
        if flag[CLASS]==2 and dis <= thres:
            true_det_count[CLASS] += 1
            false_det_count[CLASS] -= 1
            