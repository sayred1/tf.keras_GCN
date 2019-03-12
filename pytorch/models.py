import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

def readout_gg(_X, X, output_dim):
    """
    _X: final node embeddings
    X: initial node features
    """
    val1 = nn.Linear(torch.cat([_X, X], dim=2), output_dim, bias=True)
    val1 = F.sigmoid(val1)
    val2 = nn.Linear(_X, output_dim, bias=True)
    output = torch.mm(val1, val2)
    output = torch.sum(output, 1)
    output = F.relu(output)

    return output

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        #print('input', input.shape)
        #print('weight', self.weight.shape)
        #support = torch.mm(input, self.weight)
        support = torch.einsum('ijk,kl->ijl', input, self.weight)
        #print('input', input.shape)
        #print('weight', self.weight.shape)
        #print('support', support.shape)
        #print('adj', adj.shape)
        output = torch.bmm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeas, nhids):
        super(GCN, self).__init__()
        #print('in GCN')
        self.gc1 = GraphConvolution(nfeas[0], nhids[0])
        self.gc2 = GraphConvolution(nhids[0], nhids[1])
        self.gc3 = GraphConvolution(nhids[1], 1)
        self.gc4 = nn.Linear(nfeas[1], nhids[2])
        self.gc5 = nn.Linear(nhids[2], nhids[3])
        self.gc6 = nn.Linear(nhids[3], nhids[4])
        self.gc7 = nn.Linear(nhids[4], nhids[5])
        #self.dropout = dropout

    def forward(self, x, adj):
        #print('adj', adj.shape)
        #print('gc1: \n', x)
        #x = F.dropout(x, self.dropout, training=self.training)
        gconv1 = F.relu(self.gc1(x, adj))
        gconv2 = F.relu(self.gc2(gconv1, adj))
        gconv3 = F.relu(self.gc3(gconv2, adj))
        #x = x.view(x.shape[0], x.shape[1])       
        #print('gc2: \n', x.shape)
        #print('gc3: \n', x.shape)
        #x = F.relu(self.gc5(x))
        #x = F.relu(self.gc6(x))
        #x = F.relu(self.gc7(x))
        #print('gc4: \n', x.shape)
        #missing permutation invariance
        #graph_feature = readout_gg(gconv1, gconv2, 128)
        #x = F.relu(self.gc5(graph_feature))
        gconv3 = gconv3.view(gconv3.shape[0], gconv3.shape[1]) 
        #print('gc4: \n', gconv3.shape)
        x = F.relu(self.gc4(gconv3))
        x = F.relu(self.gc5(x))
        x = F.relu(self.gc6(x))
        x = F.relu(self.gc7(x))

        return x
