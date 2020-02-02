import numpy as np
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


	
 
	# return the intersection over union value

	return (minw*minh/(maxw*maxh))
