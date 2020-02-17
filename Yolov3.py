
import torch
import torch.nn as nn
import numpy as np



class Yolo(nn.Module):
    def __init__(self,cfgfile,num_classes,anchors):
        super(Yolo,self). __init__()
        self.blocks = parse_cfg(cfgfile)
        self.num_classes = num_classes
        self.net_info, self.module_list = createModules(self.blocks)

    def forward(self):
        outputs = []
        for i,(block.module) in enumerate(zip(blocks,self.module_list)):
            a = 0
