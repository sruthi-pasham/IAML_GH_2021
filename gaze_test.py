from gaze import *
from utils import *
from detector import *
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils


gaze_error_with_movement = []
gaze_error_without_movement = []
pupil_error_with_movement = []
pupil_error_without_movement = []

#Load the data
dirs = os.listdir('/home/sruthi/Desktop/IAML_GH_2021/IAML_GH_2021/inputs/images')
print(dirs)

#for each folder 
for dir in dirs:
    print(dir)
    
    #load .json file for positions 
    pos = utils.load_json(dir,"positions")
    #print(pos)
    
    #load .json file for pupils
    pupls = utils.load_json(dir,"pupils")
    #print(pupls)
    
    #load images
    img = utils.load_images(dir)
    #print(img)

    #calculate the distance
    #create and calibrate a GazeModel 
    train_size = 9
    #print(len(img))
    model = GazeModel(img[:train_size], pos[:train_size])


    cou = (len(img[train_size:]))
    #print(cou)
    
    #estimate for non calibrated images
    for i in range(cou):
        #for gaze
        x,y = model.estimate(img[train_size+i])
        #print(x,y)
        #calculate distance between estimated and ground truth values for gaze
        gaze_dist = dist(pos[train_size+i],np.array([y,x]))
        #print(gaze_dist)

        #for pupil
        #pupil ground truth values from .json
        pupil_ground_truth_vals = pupil_json_to_opencv(pupls[train_size+i])
        #pupil from detector.py
        pupil_detector = find_pupil(img[train_size+i])
        #coordinates
        ground_truth = np.array(pupil_ground_truth_vals[0])
        detected = np.array(pupil_detector[0])
        #dist between ground truth and detected
        pupil_error=dist(ground_truth,detected)

        #append with and without head movement seperately
        if(dir=="moving_medium" or dir=="moving_hard"):
            gaze_error_with_movement.append(gaze_dist)
            pupil_error_with_movement.append(pupil_error)
        else:
            gaze_error_without_movement.append(gaze_dist)
            pupil_error_without_movement.append(pupil_error)

#Data analysis
#mean gaze error without movement
mean_without_movement=np.mean(gaze_error_without_movement)
print("mean error without movement:", mean_without_movement)
#mean gaze error with movement
mean_with_movement=np.mean(gaze_error_with_movement)
print("mean error with movement:",mean_with_movement)        

#median gaze error without movement
median_without_movement=np.median(gaze_error_without_movement)
print("median error without movement:", median_without_movement)
#median gaze error with  movement
median_with_movement=np.median(gaze_error_with_movement)
print("median error with movement:", median_with_movement)

#histogram without movement
plt.figure("Histogram without head movement")
plt.hist(gaze_error_without_movement, density=True, cumulative=True)
plt.xlabel("Gaze error without movement")
plt.ylabel("Occurrence density")
plt.savefig("/home/sruthi/Desktop/gaze_error_without_movement.png")

#histogram with movement
plt.figure("Histogram with head movement")
plt.hist(gaze_error_with_movement, density=True, cumulative=True)
plt.xlabel("Gaze error with movement")
plt.ylabel("Occurrence density")
plt.savefig("/home/sruthi/Desktop/gaze_error_with_movement.png")

#Correlation betweem pupil detection distance and gaze distance errors
print("correlation coeff, without movement", np.corrcoef(pupil_error_without_movement, gaze_error_without_movement))
print("correlation coeff, with movement", np.corrcoef(pupil_error_with_movement, gaze_error_with_movement))

#Further analysis
#Precision (TP/(TP+FP))
#What proportion of positive identifications was actually correct?

#Recall (TP/(TP+FN))
#What proportion of actual positives was identified correctly?

#Intersection over Union(IOU)

#Average Precision(AP)

#Mean Average Precision(mAP)
