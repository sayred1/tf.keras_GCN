from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim

from models import GCN

def load_data(path="../data/", ids=1000):
    print('Loading {} dataset...'.format(path))
    features = np.load(path+'fea.npy')[:ids]
    adj = np.load(path+'adj.npy')[:ids]
    prop = np.load(path+'prop.npy')[:ids]
    features = torch.FloatTensor(features)
    adj = torch.FloatTensor(adj)
    prop = torch.FloatTensor(prop)

    return adj, features, prop


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-2,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Load data
adj, features, props = load_data()
#nprint(adj.shape)
#sys.exit()
# Model and optimizer

model = GCN(nfeas = [features.shape[2], features.shape[1]],
            nhids=[32, 32, 128, 128, 128, 1]
            )

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    #print(features, adj)
    #output = model(features[i].transpose(0,1), adj[i])
    output = model(features, adj)
    #print('props', props.shape)
    #loss_train = F.mse_loss(output, props)
    loss_train = F.l1_loss(output, props)
    #acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    # Evaluate validation set performance separately,
    # deactivates dropout during validation run.
    #model.eval()
    #output = model(features, adj)
    #loss_val = F.nll_loss(output, props)
    #acc_val = accuracy(output, props)
    if (epoch+1)%20 == 0:
        mse = F.mse_loss(output, props)
        print('Epoch: {:04d}  MAE: {:6.4f} MSE: {:6.4f} Time: {:6.2f}'.format(epoch+1, loss_train.item(), mse.item(), time.time() - t))


#def test():
#    model.eval()
#    output = model(features, adj)
#    loss_test = F.mse_loss(output[]
#    acc_test = accuracy(output[idx_test], labels[idx_test])
#    print("Test set results:",
#          "loss= {:.4f}".format(loss_test.item()),
#          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
#test()
