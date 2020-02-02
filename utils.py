import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.patches as patches
from PIL import Image
import matplotlib.pyplot as plt

def add_patch(ax,x_min,y_min,x_max,y_max,color):
	ax.add_patch(patches.Rectangle((x_min,y_min),np.abs(x_min-x_max),np.abs(y_max-y_min),linewidth=2,edgecolor=color,facecolor='none'))

def display_bounding_boxes(df,path_to_images=None,path_to_labels=None,specific_image=None):
	path_to_labels = "/home/mlrig/Documents/Yolo3 Implementation/data/training/label_2//"
	path_to_images = "/home/mlrig/Documents/Yolo3 Implementation/data/training/image_2//"
	"""""Display the image with the label bounding box over it,You can also specify the concrete image """
	if specific_image is None:
		data = df.loc[np.random.randint(0,df.shape[0])]
		file = data["image"]
		new_df  = df.loc[df["image"]==file]
		path = os.path.join(path_to_images,str(data["image"]))+".png"
		
	if specific_image!=None:	
		new_df = df.loc[df["image"]==specific_image]
		path = os.path.join(path_to_images,specific_image)+".png"

	im = np.array(Image.open(path), dtype=np.uint8)
	fig,ax = plt.subplots(1)
	print("path to the image",path)

	ax.imshow(im)
	for i,j in new_df.iterrows():
 			x_min = float(j["x_min"])
 			y_min = float(j["y_min"])
 			x_max = float(j["x_max"])
 			y_max  = float(j["y_max"])
 			if j["type"]=="Vehicle":
 				color = 'r'
 			elif j["type"]=="DontCare":
 				color = 'g'
 			elif j["type"]=="Pedestrian":
 			 	color = 'b'
 			elif j["type"]=="Misc":
 				color = "y"
 			elif j["type"]=="Cyclist":
 				color  = "w"
 			elif j["type"]=="Tram":
 				color = 'k'				
 			add_patch(ax,x_min,y_min,x_max,y_max,color)
	plt.show() 			

def create_label_datframe(path_to_labels=None):
	path_to_labels = "/home/mlrig/Documents/Yolo3 Implementation/data/training/label_2/"
	labels = os.listdir(path_to_labels)
	d = {"image":[],"type":[],"x_min":[],"y_min":[],"x_max":[],"y_max":[]}	
	for label in labels:
		f = open(os.path.join(path_to_labels,label), "r")
		data = f.read()
		chunk = data.split("\n")
		image_selected  = label.split(".")[0]
		for register in chunk:
			register = register.split(" ")
			if len(register) == 15:
				d["image"].append(image_selected)
				if register[0] in ['Person_sitting']:
					register[0] = "Pedestrian"
				if register[0] in ["Van","Car","Truck"]:
					register[0] = "Vehicle"
				d["type"].append(register[0])
				d["x_min"].append(register[4])
				d["y_min"].append(register[5])
				d["x_max"].append(register[6])
				d["y_max"].append(register[7])
	df = pd.DataFrame(data=d)
	return df		