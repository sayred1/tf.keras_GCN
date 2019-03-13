from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

def load_data(path="../data/", ids=10000):
    print('Loading {} dataset...'.format(path))
    features = np.load(path+'f-fea.npy')[:ids]
    adj = np.load(path+'f-adj.npy')[:ids]
    prop = np.load(path+'f-prop.npy')[:ids]
    features = torch.FloatTensor(features)
    adj = torch.FloatTensor(adj)
    prop = torch.FloatTensor(prop)

    return adj, features, prop

def get_skip_connection(_X, X):
    if _X.shape[2] != X.shape[2]:
        out_dim = _X.shape[2]
        X = nn.Linear(X.shape[2], out_dim, bias=False)(X)
    _X = F.relu(_X + X) 
    return _X

def readout_gg(_X, X, output_dim):
    """
    _X: final node embeddings
    X: initial node features
    """
    X0 = torch.cat([_X, X], dim=2)
    val1 = nn.Linear(X0.shape[-1], output_dim, bias=True)(X0)
    val1 = F.sigmoid(val1)
    val2 = nn.Linear(_X.shape[-1], output_dim, bias=True)(_X)
    output = torch.mul(val1, val2)
    output = torch.sum(output, 1)
    output = F.relu(output)
    return output

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 2. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.einsum('ijk,kl->ijl', input, self.weight)
        output = torch.bmm(adj, support)
        #output = get_skip_connection(output, input)

        return output + self.bias

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeas, nhids):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeas[0], 32)
        self.gc2 = GraphConvolution(32, 32)
        #self.gc3 = GraphConvolution(32, 32)
        self.readout = readout(32, [64, 64, 64, 1])

    def forward(self, x, adj):
        #x = F.dropout(x, self.dropout, training=self.training)
        gconv = F.relu(self.gc1(x, adj))
        #gconv = F.relu(self.gc2(gconv, adj))
        #gconv = F.relu(self.gc3(gconv, adj))
        #x = gconv3.view(gconv3.shape[0], gconv3.shape[1]) 
        return self.readout(gconv)

class readout(nn.Module):
    def __init__(self, nfeas, nhids):
        super(readout, self).__init__()
        self.nhids = nhids
        self.weight = Parameter(torch.FloatTensor(nfeas, nhids[0]))
        self.nn1 = nn.Linear(self.nhids[0], self.nhids[1])
        #self.nn2 = nn.Linear(self.nhids[1], self.nhids[2])
        self.nn3 = nn.Linear(self.nhids[2], self.nhids[3])

    def forward(self, X):
        Z = torch.einsum('ijk,kl->ijl', X, self.weight)
        Z = F.relu(Z)
        Z = torch.sigmoid(torch.sum(Z, 1))

        _Y = torch.relu(self.nn1(Z))
        #_Y = torch.tanh(self.nn2(_Y))
        _Y = self.nn3(_Y)
        return _Y
        
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-2,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.95,
                    help='Weight decay (L2 loss on parameters).')

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

adj, features, props = load_data()

model = GCN(nfeas = [features.shape[2], features.shape[1]],
            nhids=[32, 32, 128, 128, 128, 1]
            )
print(model)
for param in model.parameters():
    print(type(param.data), param.size())
# Train model
t_total = time.time()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
loss_fn = torch.nn.MSELoss(reduction='mean')

batch_size=10000
for epoch in range(args.epochs):
    model.train()
    for batch in range(int(features.shape[0]/batch_size)):
        t = time.time()
        fea0 = features[batch*batch_size:(batch+1)*batch_size]
        adj0 = adj[batch*batch_size:(batch+1)*batch_size]
        prop0 = props[batch*batch_size:(batch+1)*batch_size]
        output0 = model(fea0, adj0)
        #compute the loss
        loss_mse = loss_fn(output0, prop0)

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the Tensors it will update (which are the learnable weights
        # of the model)
        # Backward pass: compute gradient of the loss with respect to model parameters
        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.zero_grad()
        loss_mse.backward()
        optimizer.step()
    
    if (epoch+1)%1 == 0:
        #mae1 = F.l1_loss(output0, prop0)
        output = model(features, adj)
        mae = F.l1_loss(output, props)
        mse = F.mse_loss(output, props)
        print('Epoch: {:04d}  MAE: {:6.4f} MSE: {:6.4f} MSE1: {:6.4f} Time: {:6.2f}'.format(epoch+1, mae.item(), mse.item(), 
            loss_mse.item(), time.time() - t))
        #print(torch.mean(torch.pow(output-prop0, 2)))

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
output = model(features, adj)
# Testing
#test()
