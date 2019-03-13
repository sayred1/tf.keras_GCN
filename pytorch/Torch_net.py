from __future__ import division
from __future__ import print_function
from torch.utils import data
from torch.utils.data import DataLoader
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

def load_data(path="../data/", ids=1000):
    features = np.load(path+'f-fea.npy')[:ids]
    adj = np.load(path+'f-adj.npy')[:ids]
    prop = np.load(path+'f-prop.npy')[:ids]
    features = np.reshape(features, [ids, features.shape[1]*features.shape[2]])

    features = torch.FloatTensor(features)
    adj = torch.FloatTensor(adj)
    prop = torch.FloatTensor(prop)
    return adj, features, prop

class dataset(data.Dataset):
    """
    Initiating a dataset for PyTorch using
    DataLoader method.
    Args:
        xy: training dataset.
    """
    def __init__(self, xy):
        self.xy = xy
        self.len = xy.shape[0]
        self.X_train = torch.from_numpy(self.xy[:, 0:-1]).float()
        self.Y_train = torch.from_numpy(self.xy[:, [-1]]).float()

    def __getitem__(self, index):
        return self.X_train[index], self.Y_train[index]

    def __len__(self):
        return self.len

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.nn1 = nn.Linear(58*50, 64)
        self.nn2 = nn.Linear(64, 3)
        self.nn3 = nn.Linear(3, 1)

    def forward(self, x):
        x = F.relu(self.nn1(x))
        x = F.relu(self.nn2(x))
        x = self.nn3(x)
        return x

adj, features, props = load_data()

model = Net()
print(model)
for param in model.parameters():
    print(type(param.data), param.size())

optimizer = optim.Adam(model.parameters(), lr=1e-2)
#optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
loss_fn = torch.nn.MSELoss(reduction='mean')
Dset = dataset(np.column_stack((features, props)))
print(Dset[0])
train_loader = DataLoader(dataset=Dset,
                        batch_size = 200,
                        shuffle = True)

for epoch in range(500):
    model.train()
    for i, datas in enumerate(train_loader, 0):
        X_train, Y_train = datas

        y_train = model(X_train)
        loss = loss_fn(y_train, Y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
  
    output = model(features) #, adj)
    mae = F.l1_loss(output, props)
    mse = F.mse_loss(output, props)
    print('Epoch: {:04d}  MAE: {:6.4f} MSE: {:6.4f} '.format(epoch, mae.item(), mse.item()))
print("Optimization Finished!")
