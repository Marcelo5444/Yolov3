import torch.nn as nn
import torch
import numpy as np
import sys
from torch.autograd import Variable

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
        self.mse_loss = nn.MSELoss(reduction='elementwise_mean')
        self.ce_loss = nn.CrossEntropyLoss()
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
    def forward(self,prediction,img_size,num_classes):
        anchors = self.anchors
        grid_size = prediction.size(2)
        stride = img_size // grid_size
        box_attr = 5 + num_classes #(11)
        num_anchors = len(anchors)
        #permute changes the axis while view changes it completely.
        print(prediction.shape)
        print(prediction.size(0)*num_anchors*box_attr*grid_size*grid_size)
        prediction = prediction.view(prediction.size(0),num_anchors,box_attr,grid_size,grid_size)
        prediction.permute(0,1,3,4,2).contiguous()
        #we scale the anchors depending on our input size and grid.
        final_anchors = torch.FloatTensor([(a[0]/stride,a[1]/stride) for a in anchors])
        #now we sigmoid the center x and y
        x = torch.sigmoid(prediction[...,0])
        y = torch.sigmoid(prediction[...,1])
        w = prediction[...,2]
        h = prediction[...,3]
        confidence = torch.sigmoid(prediction[...,4])
        pred_class = torch.sigmoid(prediction[...,5])
        #torch.arange(-1,1,0.5) = 0.5 0 0.5 1
        #torch.repeat repeats the tensor in a given dimension
        grid_x = grid_x = torch.arange(grid_size).repeat(grid_size, 1).view([1, 1, grid_size, grid_size]).type(torch.FloatTensor)
        grid_y = torch.arange(grid_size).repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size]).type(torch.FloatTensor)
        anchor_w = final_anchors[:, 0:1].view((1, num_anchors, 1, 1))
        anchor_h = final_anchors[:, 1:2].view((1, num_anchors, 1, 1))
        pred_boxes = torch.FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
        print("done once")
        return pred_boxes

def buildTargets(pred_boxes,pred_cls,targets,anchors,num_anchors,num_classes,grid_size,threshold,img_size):
    batch_size = targets.size(0)
    mask = torch.zeros(batch_size,num_anchors,grid_size,grid_size)
    conf_mask = torch.ones(batch_size,num_anchors,grid_size,grid_size)
    tx = torch.zeros(batch_size,num_anchors,grid_size,grid_size)
    ty = torch.zeros(batch_size,num_anchors,grid_size,grid_size)
    tw = torch.zeros(batch_size,num_anchors,grid_size,grid_size)
    th = torch.zeros(batch_size,num_anchors,grid_size,grid_size)
    tconf = torch.ByteTensor(batch_size,num_anchors,grid_size,grid_size).fill_(0)
    tcls = torch.ByteTensor(batch_size,num_anchors,grid_size,grid_size,num_classes).fill_(0)
    num_ground_truth = 0
    num_correct = 0
    for batch_idx in range(batch_size):
        for target_idx in range(targets.shape[1]):
            # there is no target, continue
            if targets[batch_idx, target_idx].sum() == 0:
                continue
            num_ground_truth += 1

            # convert to position relative to bounding box
            gx = targets[batch_idx,target_idx, 1] * grid_size
            gy = targets[batch_idx,target_idx, 2] * grid_size
            gw = targets[batch_idx,target_idx, 3] * grid_size
            gh = targets[batch_idx,target_idx, 4] * grid_size

            # get grid box indices
            gi = int(gx)
            gj = int(gy)


            # shape of the gt box
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
            # shape of anchor box
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))
            # get iou
            anchor_iou = bboxIOU(gt_box, anchor_shapes, True)
            # ingore iou that is larger than some threshold
            conf_mask[batch_idx, anchor_iou > ignore_thres, gj, gi] = 0
            # best matching anchor box
            best = np.argmax(anchor_iou)
            #calculate the best iou between target and best pred box

            # ground truth box
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)
            # best pred box
            pred_box = pred_boxes[batch_idx, best, gj, gi].type(torch.FloatTensor).unsqueeze(0)
            mask[batch_idx, best, gj, gi] = 1
            conf_mask[batch_idx, best, gj, gi] = 1


            tx[batch_idx, best, gj, gi] = gx - gi
            ty[batch_idx, best, gj, gi] = gy - gj
            tw[batch_idx, best, gj, gi] = math.log(gw / anchors[best][0] + 1e-16)
            th[batch_idx, best, gj, gi] = math.log(gh / anchors[best][1] + 1e-16)

            target_label = int(targets[batch_idx, target_idx, 0])
            tcls[batch_idx, best, gj, gi, target_label] = 1
            tconf[batch_idx, best, gj, gi] = 1

            # calculate iou
            iou = bboxIOU(gt_box, pred_box, False)
            pred_label = torch.argmax(pred_cls[batch_idx, best, gj, gi])
            score = pred_conf[batch_idx, best, gj, gi]
            if iou > 0.5 and pred_label == target_label and score > 0.5:
                num_correct += 1

    return num_ground_truth, num_correct, mask, conf_mask, tx, ty, tw, th, tconf, tcls

def bboxIOU(box1, box2, x1y1x2y2):
    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # convert center to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    intersect_x1 = torch.max(b1_x1, b2_x1)
    intersect_y1 = torch.max(b1_y1, b2_y1)
    intersect_x2 = torch.min(b1_x2, b2_x2)
    intersect_y2 = torch.min(b1_y2, b2_y2)

    intersect_area = (intersect_x2 - intersect_x1 + 1) * (intersect_y2 - intersect_y1 + 1)

    # union area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = intersect_area/(b1_area+b2_area-intersect_area+1e-16)
    return iou
