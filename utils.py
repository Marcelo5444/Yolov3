import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.patches as patches
from PIL import Image
import matplotlib.pyplot as plt

def display_bounding_boxes(df,path_to_images=None,path_to_labels=None,specific_image=None):
	path_to_labels = "/home/mlrig/Documents/Yolo3 Implementation/data/training/label_2//"
	path_to_images = "/home/mlrig/Documents/Yolo3 Implementation/data/training/image_2//"
	"""""Display the image with the label bounding box over it,You can also specify the concrete image """
	if specific_image is None:
		data = df.loc[np.random.randint(0,df.shape[0])]
		path = os.path.join(path_to_images,str(data["image"]))+".png"
		print("path to the image",path)
		im = np.array(Image.open(path), dtype=np.uint8)
		fig,ax = plt.subplots(1)

# Display the image
		ax.imshow(im)
		x_min = float(data["x_min"])
		y_min =float(data["y_min"])
		x_max = float(data["x_max"]) 
		y_max = float(data["y_max"])

# Create a Rectangle patch
#rect = patches.Rectangle((float(data["bottom"]),float(data["left"])),float(data["right"]),float(data["top"]),linewidth=5)
		rect = patches.Rectangle((x_min,y_min),np.abs(x_max-x_min),np.abs(y_max-y_min),linewidth=2,edgecolor='b',facecolor='none')
# Add the patch to the Axes
		ax.add_patch(rect)

		plt.show()



	