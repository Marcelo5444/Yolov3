from __future__ import division
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
import matplotlib.pyplot as plt
from EmptyLayer import *
from DetectionLayer import *


class test_net(nn.Module):
    def __init__(self, num_layers, input_size):
        super(test_net, self).__init__()
        self.num_layers= num_layers
        self.linear_1 = nn.Linear(input_size, 5)
        self.middle = nn.ModuleList([nn.Linear(5,5) for x in range(num_layers)])
        self.output = nn.Linear(5,2)

    def forward(self, x):
        x = x.view(-1)
        fwd = nn.Sequential(self.linear_1, *self.middle, self.output)
        return fwd(x)

def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416,416))
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    return img_


def parse_cfg(cfgfile):
    """
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list

    """
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')     #store the lines in a list
    lines = [x for x in lines if len(x) > 0] #get read of the empty lines
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]


    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":               #This marks the start of a new block
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key,value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks
#    print('\n\n'.join([repr(x) for x in blocks]))

import pickle as pkl

class MaxPoolStride1(nn.Module):
    def __init__(self, kernel_size):
        super(MaxPoolStride1, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1

    def forward(self, x):
        padded_x = F.pad(x, (0,self.pad,0,self.pad), mode="replicate")
        pooled_x = nn.MaxPool2d(self.kernel_size, self.pad)(padded_x)
        return pooled_x




def create_modules(blocks):
    net_info = blocks[0]     #Captures the information about the input and pre-processing

    module_list = nn.ModuleList()

    index = 0    #indexing blocks helps with implementing route  layers (skip connections)


    prev_filters = 3

    output_filters = []

    for x in blocks:
        module = nn.Sequential()

        if (x["type"] == "net"):
            continue

        #If it's a convolutional layer
        if (x["type"] == "convolutional"):
            #Get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters= int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            #Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)

            #Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            #Check the activation.
            #It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activn)



        #If it's an upsampling layer
        #We use Bilinear2dUpsampling

        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
#            upsample = Upsample(stride)
            upsample = nn.Upsample(scale_factor = 2, mode = "nearest")
            module.add_module("upsample_{}".format(index), upsample)

        #If it is a route layer
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')

            #Start  of a route
            start = int(x["layers"][0])

            #end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0



            #Positive anotation
            if start > 0:
                start = start - index

            if end > 0:
                end = end - index


            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)



            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters= output_filters[index + start]



        #shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            from_ = int(x["from"])
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)



        #Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]


            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)



        else:
            print("Something I dunno")
            assert False


        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        index += 1


    return (net_info, module_list)
