from __future__ import division
from __future__ import print_function

from copy import deepcopy
import time
import argparse
import numpy as np
import sys

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from sklearn.metrics import mean_absolute_error, r2_score

def load_data(path="../data/", ids=8000):
    features = np.load(path+'f-fea.npy')[:ids]
    adj = np.load(path+'f-adj.npy')[:ids]
    prop = np.load(path+'f-prop.npy')[:ids]
    prop = np.reshape(prop, [len(prop), 1])
    features = torch.FloatTensor(features)
    adj = torch.FloatTensor(adj)
    prop = torch.FloatTensor(prop)

    return adj, features, prop

def get_skip_connection(_X, X):
    if _X.shape[2] != X.shape[2]:
        out_dim = _X.shape[2]
        _X = _X + nn.Linear(X.shape[2], out_dim, bias=False)(X)
    else:
        _X = X + X
    return _X

class readout(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(readout, self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_feature, out_feature), requires_grad=True)
        self.bias = Parameter(torch.FloatTensor(out_feature), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.zeros_(self.bias)
    def forward(self, X):
        Z = torch.einsum('ijk,kl->ijl', X, self.weight)
        Z = torch.relu(Z+self.bias)
        Z = torch.sigmoid(torch.sum(Z, 1))
        return Z
 
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, activation=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features), requires_grad=True)
        self.bias = Parameter(torch.FloatTensor(out_features), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.zeros_(self.bias)
        self.activation = activation

    def forward(self, input, adj):
        support = torch.einsum('ijk,kl->ijl', input, self.weight)
        output = torch.bmm(adj, support)
        output = get_skip_connection(output, input)
        return self.activation(output + self.bias)


class GCN(nn.Module):
    def __init__(self, nfeas, nhids):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(58, 32)
        self.gc2 = GraphConvolution(32, 32)
        self.gc3 = GraphConvolution(32, 32)
        self.readout = readout(32, 64)
        self.nn1 = nn.Linear(64, 64)
        self.nn2 = nn.Linear(64, 64)
        self.nn3 = nn.Linear(64, 1)

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = self.gc2(x, adj)
        x = self.gc3(x, adj)
        x = self.readout(x)
        x = F.relu(self.nn1(x))
        x = F.relu(self.nn2(x))
        x = self.nn3(x)
        return x

       
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')

args = parser.parse_args()

#np.random.seed(args.seed)
#torch.manual_seed(args.seed)

adj, features, props = load_data()

model = GCN(nfeas = [features.shape[2], features.shape[1]],
            nhids=[32, 32, 128, 128, 128, 1]
            )
#model = Net()
print(model)
for param in model.parameters():
    print(type(param.data), param.size())
# Train model
t_total = time.time()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#optimizer = optim.SGD(model.parameters(), lr=args.lr)
loss_fn = torch.nn.MSELoss(reduction='mean')

batch_size=100
for epoch in range(args.epochs):
    model.train()
    for batch in range(int(features.shape[0]/batch_size)):
        t = time.time()

        fea0 = features[batch*batch_size:(batch+1)*batch_size]
        adj0 = adj[batch*batch_size:(batch+1)*batch_size]
        prop0 = props[batch*batch_size:(batch+1)*batch_size]
        optimizer.zero_grad()

        output0 = model(fea0, adj0)
        loss = loss_fn(output0, prop0)
        loss.backward()
        optimizer.step()
  
    output = model(features, adj)
    output = output.data.numpy()
    prop = props.data.numpy()
    r2_train = 'r2: {:.4f}'.format(r2_score(output, prop))
    mae_train = 'mae: {:8.4f}'.format(mean_absolute_error(output, prop))
    print('Epoch:', epoch, r2_train, mae_train, loss.item())

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
output = model(features, adj).data.numpy()
prop = props.data.numpy()

import matplotlib.pyplot as plt
plt.tight_layout()
plt.scatter(prop, output)
plt.xlabel('Y0')
plt.ylabel('NN')
plt.show()

#output = model(features) #, adj)
# Testing
#test()
