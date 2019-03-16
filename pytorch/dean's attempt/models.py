import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import sys
import mechs



class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features # input feature size
        self.out_features = out_features # output feature size
        self.weight = Parameter(torch.FloatTensor(in_features, out_features)) # weight size, initially padded with zeros
        self.bias = Parameter(torch.FloatTensor(out_features))
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight = nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            if self.bias.size()!=torch.Size([1]):
                self.bias = Parameter(self.bias.repeat(50,1,1))
                self.bias = Parameter(self.bias.reshape(50,32,))
            self.bias = nn.init.xavier_uniform_(self.bias)

    def forward(self, input, adj):

        # output is new, input is old
        support = torch.einsum('ijk,kl->ijl', input, self.weight)
        output = torch.bmm(adj, support)
        output = mechs.skip(output,input)

        if self.bias is not None:
            output += output + self.bias
        else:
            output = output
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module): # called from train.py then calls GraphConvolution
    def __init__(self, dropout, nfeas, nhids):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeas[0], nhids[0]) #this needs to be propogated forward, but it isnt.
        self.gc2 = GraphConvolution(nhids[0], nhids[1])
        self.gc3 = GraphConvolution(nhids[1], nhids[1])
        #self.dropout = dropout
        #self.gc3 = GraphConvolution(nhids[1], 1)
        # we have three conv. layers above which need to be activated

        # y = xA^T + b
        self.gc4 = nn.Linear(nfeas[1], nhids[2])
        self.gc5 = nn.Linear(nhids[2], nhids[3])
        self.gc6 = nn.Linear(nhids[3], nhids[4])
        self.gc7 = nn.Linear(nhids[4], nhids[5])
        #self.dropout = dropout

    # def readout_gg(_X, X, output_dim = 128):


    def forward(self, x, adj):
        gconv1 = F.relu(self.gc1(x, adj)) # input
        #gconv1 = F.dropout(gconv1, self.dropout)
        gconv2 = F.relu(self.gc2(gconv1, adj)) # 32
        #gconv2 = F.dropout(gconv2, self.dropout)
        gconv3 = F.relu(self.gc3(gconv2, adj))
        #gconv3 = gconv3.view(gconv3.shape[0], gconv3.shape[1])
        gconv3 = mechs.gg(gconv3,gconv1,512)

        '''
        activation
        # self.gc3 = GraphConvolution(nhids[1], nhids[1]) in __init__
        # gconv3 = F.relu(self.gc3(gconv2, adj))
        # gconv3 = mechs.gg(gconv3,gconv1,512)
        # or gconv3 = mechs.atomwise(gconv3,512)
        # get rid of self.gc4
        # put gconv3 into self.gc5
        '''

        '''
        no gg
        # self.gc3 = GraphConvolution(nhids[1], 1) in __init__
        # gconv3 = F.relu(self.gc3(gconv2, adj))
        # gconv3 = gconv3.view(gconv3.shape[0], gconv3.shape[1])
        # x = F.relu(self.gc4(gconv3))
        '''
        #x = F.relu(self.gc4(gconv3)) # [batch_size, 512] #gconv here must be
        x = F.relu(self.gc5(gconv3)) # [batch, 512]
        x = F.relu(self.gc6(x)) # [batch, 512]
        x = F.relu(self.gc7(x)) # [batch, 1]
        return x
