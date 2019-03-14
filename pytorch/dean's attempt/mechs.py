import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import sys

def skip(_x,x):
    if(_x.size()[2] != x.size()[2]):
        in_dim = int(x.size()[2])
        out_dim = int(_x.size()[2])
        _x = F.relu(_x + nn.Linear(in_dim,out_dim,False)(x))
    else:
        _x = F.relu(_x+x)
    return(_x)

def gg(_x,x,output_dim = 512):
    output_dim = 512
    con_cat = torch.cat([_x, x], dim=2)
    input_dim  = int(con_cat.size()[2])
    val1 = nn.Linear(input_dim, output_dim, bias=True)(con_cat)
    val1 = F.relu(val1)
    val1 = F.sigmoid(val1)
    input_dim = int(_x.size()[2])
    val2 = nn.Linear(input_dim, output_dim, bias=True)(_x)
    output = torch.mul(val1, val2)
    output = torch.sum(output, 1)
    return(F.relu(output))

def atomwise(_x, output_dim = 512):
    input_dim = int(_x.size()[2])
    output = nn.Linear(input_dim, output_dim, True)(_x)
    output = torch.sum(output, 1)
    return(F.relu(output))
