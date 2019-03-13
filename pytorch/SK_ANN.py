from __future__ import division
from __future__ import print_function

import numpy as np
import sys
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler  

def load_data(path="../data/", ids=1000):
    features = np.load(path+'f-fea.npy')[:ids]
    adj = np.load(path+'f-adj.npy')[:ids]
    prop = np.load(path+'f-prop.npy')[:ids]
    features = np.reshape(features, [ids, features.shape[1]*features.shape[2]])
    return adj, features, prop

adj, X, y = load_data()

#scaler = StandardScaler()  
#scaler.fit(X)  
#X_0 = scaler.transform(X)  
X_0 = X

# Fitting
clf = MLPRegressor(hidden_layer_sizes=(64,3), max_iter=1500, random_state=1, verbose=True)
clf.fit(X_0, y)
print(clf)
r2_train = 'r2: {:.4f}'.format(r2_score(y, clf.predict(X_0)))
mae_train = 'mae: {:8.4f}'.format(mean_absolute_error(y, clf.predict(X_0)))
print('Training     ', r2_train, mae_train)
print("Optimization Finished!")
for i in range(len(clf.coefs_)):
    print(i, clf.coefs_[i].shape, clf.intercepts_[i].shape)
