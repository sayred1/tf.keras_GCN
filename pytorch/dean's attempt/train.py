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

import torch.optim.lr_scheduler

def load_data(path="/Users/b_eebs/tf-keras/data/", ids=10000):
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
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.95,
                    help='Weight decay (L2 loss on parameters).')

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Load data
# take this, and train it over batches of 1000 with batch_size
# equal to 10
adj, features, props = load_data()

# i've attempted SGD and Adam optimizers, I've attempted to implement a
# decay rate for the learning rate, instead of training every epoch I've trained
# over mini batches of batch size 100 w/ 100 molecules in each, atomwise is still giving
# me issuse, graph gatherer is all that works for me, in models.py i've set the reset_parameters
# method as the location where I assign an xavier initializer to the weight and bias
# dropout doesn't fix the problem. also reshaped the bias earlier.

dropout = 0.5
model = GCN(dropout, nfeas = [features.shape[2], features.shape[1]],
            nhids=[32, 32, 512, 512, 512, 1]) # this is two conv layers

def train(num_epochs,num_files,num_batches,batch_size,lr):
    loss_fn = torch.nn.MSELoss(reduction='mean')
    start_time = time.time()
    total_iter = 0
    for epoch in range(num_epochs):
        lr = lr * (0.95 ** epoch) # learning rate decay
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = 0.95)
        model.train()
        for file in range(num_files):
            for _iter in range(num_batches):
                total_iter += 1
                # prediction
                # batching
                feature_batch = (features[_iter*batch_size:
                (_iter+1)*batch_size])
                adj_batch = (adj[_iter*batch_size:
                (_iter+1)*batch_size])
                props_batch = (props[_iter*batch_size:
                (_iter+1)*batch_size])
                props_batch = props_batch.reshape(batch_size,1) # reshaped this
                # optimization
                optimizer.zero_grad()
                output = model(feature_batch, adj_batch)
                loss_mse = loss_fn(output,props_batch) # the loss is being calculated
                # correct based off of the below evaluation.
                mean = torch.mean((output-props_batch)**2) # same as above
                #print('total_iter ', total_iter, 'loss: ',loss_mse)
                loss_mse.backward()
                optimizer.step()
                # print out mae and mse every batch (every 100 iterations)
                if total_iter % 100 == 0:
                    mae = F.l1_loss(output, props_batch)
``                    mse = F.mse_loss(output, props_batch)
                    print('batch', total_iter//100, 'complete  mae =',mae, 'mse =',mse)

#def test():
#    model.eval()
#    output = model(features, adj)
#    loss_test = F.mse_loss(output[]
#    acc_test = accuracy(output[idx_test], labels[idx_test])
#    print("Test set results:",
#          "loss= {:.4f}".format(loss_test.item()),
#          "accuracy= {:.4f}".format(acc_test.item()))


# Train model

#for epoch in range(args.epochs):
#    train(epoch)
num_batches = 100
num_files = 1
batch_size = 100
train(args.epochs,num_batches,num_files,batch_size,args.lr)

#print("Optimization Finished!")
#print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
#test()
