import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.patches as patches
from PIL import Image
import matplotlib.pyplot as plt
from kmeans_Anchors import *

def add_patch(ax,x_min,y_min,x_max,y_max,color):
	ax.add_patch(patches.Rectangle((x_min,y_min),np.abs(x_min-x_max),np.abs(y_max-y_min),linewidth=2,edgecolor=color,facecolor='none'))

def plot_centered_Anchor(axis,centered,width,height,color):

		x = centered[0]-width/2
		y = centered[1]-height/2
		add_patch(axis,x,y,x+width,y+height,color)
def display_bounding_boxes(df,path_to_images=None,path_to_labels=None,specific_image=None):
	if path_to_labels == None:
		path_to_labels = "/home/mlrig/Documents/Yolo3 Implementation/data/training/label_2//"
	if path_to_images == None:
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
			x_min = j["x_min"]
			y_min = j["y_min"]
			x_max = j["x_max"]
			y_max  = j["y_max"]
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

def display_bounding_boxes_with_anchors(df,anchors_array,path_to_images=None,path_to_labels=None,specific_image=None):
	if path_to_labels == None:
		path_to_labels = "/home/mlrig/Documents/Yolo3 Implementation/data/training/label_2//"
	if path_to_images == None:
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
	score_group = 0.0
	final_group = 0.0
	for i,j in new_df.iterrows():
			x_min = j["x_min"]
			y_min = j["y_min"]
			x_max = j["x_max"]
			y_max  = j["y_max"]
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
			final_group = 0
			score_group  = 0

			for anchor_iter in range(0,anchors_array.shape[0]):
				score = IoU_fixed(j["width"],j["height"],anchors_array[anchor_iter][0],anchors_array[anchor_iter][1])
				if score > score_group:
					final_group = anchor_iter
			center = (x_min+0.5*j["width"],y_min+0.5*j["height"])
			print(anchors_array[final_group][0])
			plot_centered_Anchor(ax,center,j["width"],j["height"],"b")
			plot_centered_Anchor(ax,center,anchors_array[final_group][0],anchors_array[final_group][1],"r")		
plt.show()


def create_label_datframe(path_to_labels=None):
	if path_to_labels == None:
		path_to_labels = "/home/mlrig/Documents/Yolo3 Implementation/data/training/label_2//"
	labels = os.listdir(path_to_labels)
	d = {"image":[],"type":[],"x_min":[],"y_min":[],"x_max":[],"y_max":[],"height":[],"width":[]}
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
				d["x_min"].append(float(register[4]))
				d["y_min"].append(float(register[5]))
				d["x_max"].append(float(register[6]))
				d["y_max"].append(float(register[7]))
				width  = np.abs(float(register[4])-float(register[6]))
				d["width"].append(width)
				heigth  = np.abs(float(register[5])-float(register[7]))
				d["height"].append(heigth)
	df = pd.DataFrame(data=d)
	return df
