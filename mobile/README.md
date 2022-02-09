# M2 Internship
### USTH-M2-2020-2021


**Author:** 
* HUYNH Vinh Nam [M19.ICT.007]

## Steps
### 1. Install Anaconda (follow the outer README.md file)
### 2. Set up an environment
- Create an environment with python 3.7:
```
$ conda create --name mobile python=3.7
$ conda activate mobile
```

- Install the library:
```
(mobile) $ pip install tflite-model-maker
(mobile) $ pip install pycocotools
```

- Run the training:
```
(mobile) $ python model_maker_object_detection_tflite.py
```

- In file ```model_maker_object_detection_tflite.py```:
  - Line 15 allows you to config the training model [efficientdet_lite0, efficientdet_lite1, efficientdet_lite2, efficientdet_lite3, efficientdet_lite4].
  - Line 17 must be the correct directory of the ```data.csv``` file.

- Testing the trained model:  
```
(mobile) $ python test_mobile_inference.py
```
- In file ```test_mobile_inference.py```:
  - Line 21, please specify the trained model (Here I provided 3 trained models: ```model-lite0.tflite```, ```model-lite3.tflite```, ```model-lite4.tflite```).
  - Line 131, please specify the image file.
