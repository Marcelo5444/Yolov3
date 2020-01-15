#!/usr/bin/env python
# coding: utf-8

# In[3]:


from __future__ import division
print("hello from the yolov3")

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np


# In[4]:


def parse_cfg(cfgfile):
	"""
	Takes a configuration file
	
	Returns a list of blocks. Each blocks describes a block in the neural
	network to be built. Block is represented as a dictionary in the list
	
	"""
	file = open(cfgfile, 'r')
	lines = file.read().split('\n')                        # store the lines in a list
	lines = [x for x in lines if len(x) > 0]               # get read of the empty lines 
	lines = [x for x in lines if x[0] != '#']              # get rid of comments
	lines = [x.rstrip().lstrip() for x in lines]
	block = {}
	blocks = []
	for line in lines:
		if line[0] =='[':
			if len(block) != 0:
				blocks.append(block)
				block = {}
			block["type"] = line[1:-1].rstrip()
		else:
			key,value = line.split('=')
			block[key.rstrip()] = value.lstrip()
	blocks.append(block)	
	return blocks
def create_modules(blocks):
	net_info = blocks[0]
	module_list = nn.ModuleList()
	prev_filters = 3
	output_filters = []	

			
        
                
                


# In[6]:


blok = parse_cfg("./cfg/yolov3.cfg")
print(blok)    


# In[ ]:




