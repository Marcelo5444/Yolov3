import numpy as np
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.patches as patches
from PIL import Image
import matplotlib.pyplot as plt
from utils import *

def IoU(boxA,boxB):
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou
def IoU_fixed(w1,h1,w2,h2):
	if h1 >= h2:
	 	maxh = h1
	 	minh = h2
	else:
	 	maxh = h2
	 	minh = h1
	if w1 >= w2:
	   maxw = w1
	   minw = w2
	else:
		maxw  = w2
		minw = w1
	return (minw*minh/(maxw*maxh))

def update_anchors(anchors,data):
    for i in range(0,anchors.shape[0]):
        if len(data[data[:,2]==i])!=0:

            mean = np.mean(data[data[:,2]==i],axis=0)[:2]
            for item in mean:
                if np.isnan(item) or item <0:
                    mean[item] = anchors[i,item]             
            anchors[i,:] = mean

    return anchors

def kmeans_anchors(data,iterations=10,num_anchors=4,resolution=(1200,400)):
    width = np.random.randint(low=0,high=1200,size=num_anchors)
    height = np.random.randint(low=0,high=400,size=num_anchors)
    anchors = np.column_stack((width,height))
    sum_of_scores = 0.0
    for i in range(0,iterations):
        for iterator in range(0,data.shape[0]):
            score_group = 0.0
            
            final_group = 0.0
            for anchor_iter in range(0,num_anchors):
                score = IoU_fixed(data[iterator][0],data[iterator][1],anchors[anchor_iter][0],anchors[anchor_iter][1])                 
            
                if score > score_group:
                    final_group = anchor_iter
                    score_group  = score
                if i == iterations-1:
                    sum_of_scores = score_group + sum_of_scores
            data[iterator][2] = final_group     
        
        anchors = update_anchors(anchors,data)
        display_anchors(anchors)


    return anchors

def display_anchors(anchors):
    lims = (0, 500)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')
    for i in range(0,anchors.shape[0]):
        ax1.add_patch(
        patches.Rectangle((0, 0), anchors[i,0], anchors[i,1],fill=False))
    plt.ylim(lims)
    plt.xlim(lims)
    

 def display_Anchors_and_BB()       
    
    
    
        