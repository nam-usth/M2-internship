# M2 Internship
### USTH-M2-2020-2021


**Author:** 
* HUYNH Vinh Nam [M19.ICT.007]

**Topic:**
* Automatic landmarks detection on Insect wings using Deep Learning method

## Brief Summary
This repository provides two possible approaches to detect landmarks automatically by using: 
1. Tensorflow Object Detection API models [Faster R-CNN, SSD] or a customized YOLOv3 model
2. Regression models [adapted some EfficientNet models (B0/B3/EffNetV2) + transfer learning]

I've been able to have this repository successfully deployed in both Windows 10 and Linux-based OSes. 
The Tensorflow version used in my project is currently 2.3.0, CUDA 10.1.243 + CuDNN 7.6.5.

For using my pre-trained detection model on mobile application, I've also worked on the tflite model maker (which requires tf 2.4.0 with CUDA 11.2 + CuDNN 8.2.1 for training and export the inference graph). 
However, this repository mainly focused on running the inference in PC, hence, let's set this tflite model aside.

[Appendix: Known issues](https://github.com/Protossnam/M2_internship#appendix-known-issues)

## Steps
### 1. Install Anaconda
* On Windows: You can find the download link of Anaconda from [here](https://www.anaconda.com/products/individual). Once it's downloaded, execute the installer file and work through the installation steps.
* On Linux:
1. Just simply open your terminal and type:
```
$ cd /tmp
$ curl -O https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
$ sha256sum Anaconda3-2020.02-Linux-x86_64.sh
$ bash ~/Downloads/Anaconda3-2020.02-Linux-x86_64.sh
```
2. You can keep pressing ENTER until the end of the license agreement. 

Once you agree to the license, you will be prompted to choose the location of the installation. 

3. You can press ENTER to accept the default location. 

Once installation is complete, the following output should show up:

```
Output
...
installation finished.
Do you wish the installer to prepend the Anaconda3 install location
to PATH in your /home/namhv/.bashrc ? [yes|no]
[no] >>> 
```
Please type ```yes``` to use the ```conda``` command.  

4. After this step, it's time to activate the installation:
```
$ source ~/.bashrc
```

### 2. Set up an environment
* Congratulation, you have the Anaconda ready on your machine, now is the time to clone this repository.
Rename “M2_internship-main” to just “M2_internship”.
* On both Windows and Linux, from the Anaconda Prompt/Terminal:
```
$ conda create --name insect_wing python=3.7
$ conda activate insect_wing
```

* You should change the current working directory to the cloned folder [M2_internship], then run:
```
(insect_wing) $ conda install --yes --file requirements.txt
```

### 3. User manual

#### 3a. Detect landmarks by using Object Detection method
##### In the object_detection folder:
###### Using some pre-trained Tensorflow Object Detection API models
- Running the detection: in ```Object_detection_image.py```:
  - Line 35 allows you to choose which model to run the inference (Faster R-CNN or SSD Mobilenet)
  - Line 36 is an image from the images/test folder (if you pay a little attention, you will notice that the Line 49 will point to the test images folder).  
  Change this filename whenever you want to inference another test image.

- I provided 2 tfrecord files: the ```train.record```, the ```test.record``` and the dataset with .csv files. 
Therefore, it will be easier for anyone who wants to re-train or reproduce my result.

###### Using a custom pre-trained YOLOv3-tiny model
- Running the detection: navigate to the subfolder yolo_pre-trained, you will find the ```yolo.py```:
  - Line 9, please change this to the correct absolute directory, where you cloned my repository.  
  **_E.g._** "D:/M2_internship/object_detection/images/test"
  
#### 3b. Detect landmarks with pre-trained Regression models
##### In the regression folder:
- File ```regression_model.py``` allows:
  - Training the regression network:
  
  - Predict landmarks from trained network:
  
- File ```evaluation.py``` allows:
  - Compute the difference between predicted result with groundtruth landmarks.

## Appendix: Known issues

#### 1. PackagesNotFoundError: The following packages are not available from current channels:
```
  - tensorflow-gpu-estimator==2.3.0
  - tensorboard==2.4.1
  - opencv-python==4.2.0.34
  - slim==0.5.9
  - utils==1.0.1
  - protobuf==3.12.2
```

To fix this, open the requirements.txt file, please kindly remove these 6 lines of package. 
We will use ```pip install``` command to get them later.

#### 2. Corrupted .pb inference graph file when cloning this repository.
```
...
Traceback (most recent call last):
  File "Object_detection_image.py", line 69, in <module>
    od_graph_def.ParseFromString(serialized_graph)
google.protobuf.message.DecodeError: Error parsing message
```

I figured this problem when deployed my work to USTH ICTLab server. The corrupted file has only 134 B which is completely wrong!
In order to get the correct file, please download the frozen_inference_graph.pb file from my repository.

#### 3. Error while training with RTX 30 series
I have been reported that the new RTX 30 series comes with the latest CUDA 11 and CuDNN 8 and only compatible with the Tensorflow >= 2.4.0. So for all the older GPU (date-back to the RTX 20 series), it is recommended to follow the ```requirements.txt```. For those who has the RTX 30 series, we need to modify the requirements a little bit:

You need:
```
  - CUDA 11.2 or higher
  - CuDNN 8.2.1 (since this hasn't existed on Anaconda yet, you have to install it directly from Nvidia)
  - Tensorflow >= 2.4.0
```
