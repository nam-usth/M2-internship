import cv2
import os
import xml.etree.ElementTree as ET
from utils import label_map_util
from utils import visualization_utils as vis_util
import tensorflow.compat.v1 as tf
import numpy as np
import math
import pandas as pd

PATH_TO_CKPT = os.path.join('D:/USTH_Master/Internship/models/research/object_detection_insect_wing/inference_graph/frozen_inference_graph.pb')

PATH_TO_LABELS = os.path.join('D:/USTH_Master/Internship/models/research/object_detection_insect_wing/training/insect_wing_object.pbtxt')
NUM_CLASSES = 15

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

lm9 = []


for file in os.listdir("D:/USTH_Master/Internship/models/research/object_detection_insect_wing/images/test/")[0:3]:
    if(file[-3:]=="xml"):
        link = os.path.join("D:/USTH_Master/Internship/models/research/object_detection_insect_wing/images/test/", file)
        print(file)
        tmp = cv2.imread(link)
        treee = ET.parse(link)
        root = treee.getroot()
        xmin=int(root[14][4][0].text)
        ymin =int(root[14][4][1].text)
        xmax=int(root[14][4][2].text)
        ymax=int(root[14][4][3].text)
        print(xmin, ymin, xmax, ymax)
        MODEL_NAME = 'inference_graph'
        IMAGE_NAME = file[:-4]+'.jpg'

        PATH_TO_IMAGE = link = os.path.join("D:/USTH_Master/Internship/models/research/object_detection_insect_wing/images/test/", IMAGE_NAME)

        image = cv2.imread(PATH_TO_IMAGE)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_expanded = np.expand_dims(image_rgb, axis=0)
        height = image.shape[0]
        width = image.shape[1]
        
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})
        all_boxes, all_classes, all_scores = [], [], []
        
        max_score = -100000
        xmi, ymi, xma, yma = -1, -1, -1, -1
        htmp = -1
        for i in range(0,int(num[0])-1):
            if(i==0):
                print(classes[0][i])
                print(max_score<scores[0][i])
            if(int(classes[0][i])==9)and(max_score<scores[0][i]):
                htmp=i
                ymi = int(boxes[0][i][0]*height)
                xmi = int(boxes[0][i][1]*width)
                yma = int(boxes[0][i][2]*height)
                xma = int(boxes[0][i][3]*width)
        print(htmp)
        print(xmi,ymi,xma,yma)
        rx_true = xmin+(xmax-xmin)/2
        ry_true = ymin+(ymax-ymin)/2
        rx_pred = xmi+(xma-xmi)/2
        ry_pred = ymi+(yma-ymi)/2
        dis = math.sqrt((rx_true-rx_pred)*(rx_true-rx_pred)+(ry_true-ry_pred)*(ry_true-ry_pred))
        lm9.append([rx_true, ry_true, rx_pred, ry_pred, dis])
        print(classes)
        print(scores)
        
df_lm9 = pd.DataFrame(lm9)
df_lm9.to_csv('D:/USTH_Master/Internship/models/research/object_detection_insect_wing/lm9.csv')
            
            
