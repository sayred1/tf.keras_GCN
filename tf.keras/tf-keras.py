import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error, r2_score

from tensorflow.keras import backend as K
K.set_floatx('float64')

class GCN(layers.Layer):
    """
    A graph convolution model
    """
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(GCN, self).__init__(**kwargs)

    def build(self, input_shape):
        #print(input_shape[0][2])
        shape = tf.TensorShape((input_shape[0][2], self.output_dim))
        shape = [int(shape[0]),int(shape[1])] # [50 , 32]

        self.kernel = self.add_weight(name='kernel',
                                      shape=shape,
                                      initializer='glorot_uniform',
                                      #initializer='he_uniform',
                                      trainable=True,
                                      dtype=tf.float64)

        self.bias = self.add_weight(name='bias',
                                    shape=[shape[1]],
                                    initializer='glorot_uniform',
                                    #initializer='he_uniform',
                                    trainable=True,
                                    dtype=tf.float64)

    def call(self, input):
        X, A = input[0], input[1]
        dim = self.kernel.get_shape()[1]
        num_atoms = A.get_shape()[1]
        _b = tf.reshape(tf.tile(self.bias, [num_atoms]), [num_atoms, dim])
        _X = tf.einsum('ijk,kl->ijl', X, self.kernel) + _b
        _X = tf.matmul(A, _X)
        _X = get_skip_connection(_X, X)
        return _X

class G2N(layers.Layer):
    """
    A layer to sum the node feature to preserve the permutation invariance
    """
    def __init__(self, output_dim,  **kwargs):
        self.output_dim = output_dim
        super(G2N, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[2], self.output_dim))
        shape = [int(shape[0]),int(shape[1])] # [50 , 32]

        self.kernel = self.add_weight(name='kernel',
                                      shape=shape,
                                      #initializer='he_uniform',
                                      initializer='glorot_uniform',
                                      trainable=True,
                                      dtype=tf.float64)
        self.bias = self.add_weight(name='bias',
                                    shape=[shape[1]],
                                    #initializer='he_uniform',
                                    initializer='glorot_uniform',
                                    trainable=True,
                                    dtype=tf.float64)


    def call(self, X):
        dim = self.kernel.get_shape()[1]
        num_atoms = X.get_shape()[1]
        _b = tf.reshape(tf.tile(self.bias, [num_atoms]), [num_atoms, dim])
        Z = tf.einsum('ijk,kl->ijl', X, self.kernel)
        Z += _b
        Z = tf.nn.relu(Z)
        Z = tf.nn.sigmoid(tf.reduce_sum(Z, 1))
        return Z

class mymodel(tf.keras.Model):
    """
    A generallized model for the prediction of properties from graph
    """
    def __init__(self, model_name='GCN', graph_layers=[32, 32], mlp_layers=[128, 128], **kwargs):
        super(mymodel, self).__init__(**kwargs)
        self.model_name = model_name
        self.graph_layers = []
        self.mlp_layers = []
        if model_name == 'GCN':
            for hid in graph_layers:
                self.graph_layers.append(GCN(hid)) 
        self.g2n = G2N(graph_layers[-1])
        for mlp in mlp_layers:
            self.mlp_layers.append(layers.Dense(128, activation=tf.nn.relu, input_shape=[mlp]))
        self.mlp_layers.append(layers.Dense(1))

    def call(self, input):
        x, A = input[0], input[1]
        for model in self.graph_layers:
            x = tf.nn.relu(model([x, A]))
        x = tf.nn.relu(self.g2n(x))
        for model in self.mlp_layers:
            x = model(x)
        return x

def get_skip_connection(_X, X):
    if( int(_X.get_shape()[2]) != int(X.get_shape()[2]) ):
       out_dim = int(_X.get_shape()[2])
       _X = tf.nn.relu(_X + tf.layers.dense(X, units = out_dim, use_bias=False))
    else:
       _X = tf.nn.relu(_X + X) 

    return _X 

def load_data(id1, id2, unit=10000):
    """
    load the source data from somewhere
    """
    features = np.load('../../augmented-GCN/database/ZINC/features/'+str(id1)+'.npy')
    adj = np.load('../../augmented-GCN/database/ZINC/adj/'+str(id1)+'.npy')
    
    for i in range(id1+1, id2):
        fea0 = np.load('../../augmented-GCN/database/ZINC/features/'+str(i)+'.npy')
        adj0 = np.load('../../augmented-GCN/database/ZINC/adj/'+str(i)+'.npy')
        features = np.concatenate((features, fea0), axis=0) 
        adj = np.concatenate((adj, adj0), axis=0)
    N_sample = len(features)
    prop = np.load('../../augmented-GCN/database/ZINC/logP.npy')[id1*unit:id2*unit]
    return adj, features, prop

class Progress(keras.callbacks.Callback):
    """
    A simple function to show the progress
    """
    def on_epoch_end(self, epoch, logs):
        if epoch % 1 == 0:
             print('epoch-----', epoch, logs)


#load data
adj1, features1, props1 = load_data(0, 40)
adj2, features2, props2 = load_data(40,49)

#Define the model
lr = 1e-3
model = mymodel(model_name='GCN', graph_layers=[32, 32], mlp_layers=[128, 64])
#optimizer = tf.keras.optimizers.RMSprop(0.001)
optimizer = tf.keras.optimizers.Adam(lr=lr, decay=1e-6)
model.compile(loss='mean_squared_error',
              optimizer=optimizer,
              metrics=['mean_absolute_error'])
history = model.fit(x=[features1, adj1], y=props1, batch_size=100, 
                    epochs=100, verbose=0,
                    callbacks=[Progress()]
                   )

#history1 = model.evaluate(x=[features2, adj2], y=props2, batch_size=100)

#Analyze the results
res1 = model.predict([features1,adj1])
res2 = model.predict([features2,adj2])
print('r2 in train: ', r2_score(res1, props1))
print('r2 in test: ', r2_score(res2, props2))

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
plt.figure(figsize=(10, 8))
plt.subplot(211)
plt.title(model.model_name+'-'+str(len(model.graph_layers))+'-'+str(len(model.mlp_layers)))
plt.xlabel('Epoch')
plt.ylabel('Mean Abs Error')
plt.plot(hist['epoch'], hist['mean_absolute_error'],
         label='Train Error')
#plt.plot(hist['epoch'], hist['mean_absolute_error'],
#         label = 'Val Error')
plt.legend()

plt.subplot(212)

label1 = 'Train: r2: {:.4f} mae: {:.4f} in {:d}'.format(r2_score(res1, props1), 
                                         mean_absolute_error(res1, props1),
                                                              len(props1))
label2 = 'test:  r2: {:.4f} mae: {:.4f} in {:d}'.format(r2_score(res2, props2), 
                                         mean_absolute_error(res2, props2),
                                                              len(props2))
plt.scatter(props1, res1, label=label1)
plt.scatter(props2, res2, label=label2)
plt.legend()
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.savefig('res.png')

print(model.summary())

# Recreate the exact same model, including weights and optimizer.
#model.save('my_model.h5')
#new_model = keras.models.load_model('my_model.h5')
#new_model.summary()
