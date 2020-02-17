
import torch
import torch.nn as nn
import numpy as np
"https://github.com/ultralytics/yolov3/blob/master/models.py"
"https://github.com/Ray-Luo/YOLOV3-PyTorch/blob/master/model/YOLO.py"
"http://leiluoray.com/2018/11/10/Implementing-YOLOV3-Using-PyTorch/#how-anchor-boxes-work"



class Yolo(nn.Module):
    def __init__(self,cfgfile,num_classes,anchors):
        super(Yolo,self). __init__()
        self.blocks = parse_cfg(cfgfile)
        self.num_classes = num_classes
        self.net_info, self.module_list = createModules(self.blocks)

    def forward(self,x):
        outputs = []
        layer_outputs = []
        blocks = self.blocks[1:]
        for i,(block,module) in enumerate(zip(blocks,self.module_list)):
            if block["type"] in ["convolutional",'upsample']:
                x = module(x)
            elif block["type"] == "route":
                #we obatin all the route layers we later concat data from.
                layer_i = [int(x) for x in block["layers"]]
                try:
                    x = torch.cat
                except:
                    print("size missmatch")
