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
