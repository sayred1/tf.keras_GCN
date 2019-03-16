import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error, r2_score

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
                                  trainable=True,dtype=tf.float32)

        self.bias = self.add_weight(name='bias',
                        shape=[shape[1]],
                        initializer='glorot_uniform',
                        trainable=True,dtype=tf.float32)

    def call(self, input):
        X, A = input[0], input[1]
        dim = self.kernel.get_shape()[1]
        num_atoms = A.get_shape()[1]
        _b = tf.reshape(tf.tile(self.bias, [num_atoms]), [num_atoms, dim])
        _X = tf.einsum('ijk,kl->ijl', X, self.kernel) + _b
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
                                  initializer='glorot_uniform',
                                  trainable=True,dtype=tf.float32)
        self.bias = self.add_weight(name='bias',
                        shape=[shape[1]],
                        initializer='glorot_uniform',
                        trainable=True,dtype=tf.float32)


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
    def __init__(self):
        super(mymodel, self).__init__(name='')
        self.gcn1 = GCN(32) #, input_shape=[58])
        #self.gcn2 = GCN(32) #, input_shape=[58])
        self.gcn3 = GCN(32) #, input_shape=[58])
        self.g2n = G2N(512)#, input_shape=[58])
        self.dense1 = layers.Dense(512, activation=tf.nn.relu, input_shape=[512])
        self.dense2 = layers.Dense(512, activation=tf.nn.tanh, input_shape=[512])
        self.dense3 = layers.Dense(1)

    def call(self, input):
        X, A = input[0], input[1]
        x = tf.nn.relu(self.gcn1([X, A]))
        #x = tf.nn.relu(self.gcn2([x, A]))
        x = tf.nn.relu(self.gcn3([x, A]))
        x = tf.nn.relu(self.g2n(x))
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)

def get_skip_connection(_X, X):
    if( int(_X.get_shape()[2]) != int(X.get_shape()[2]) ):
       out_dim = int(_X.get_shape()[2])
       _X = tf.nn.relu(_X + tf.layers.dense(X, units = out_dim, use_bias=False))
    else:
       _X = tf.nn.relu(_X + X) 

    return _X 

def load_data():
    """
    load the source data from somewhere
    """
    path = '../data/'
    #features = np.load(path+'f-fea.npy')
    #adj = np.load(path+'f-adj.npy')
    features = np.load('../../augmented-GCN/database/ZINC/features/0.npy')
    adj = np.load('../../augmented-GCN/database/ZINC/adj/0.npy')
    #prop = np.load(path+'f-prop.npy')
    for i in range(1, 10):
        fea0 = np.load('../../augmented-GCN/database/ZINC/features/'+str(i)+'.npy')
        adj0 = np.load('../../augmented-GCN/database/ZINC/adj/'+str(i)+'.npy')
        features = np.concatenate((features, fea0), axis=0) 
        adj = np.concatenate((adj, adj0), axis=0)
    N_sample = len(features)
    prop = np.load('../../augmented-GCN/database/ZINC/logP.npy')[:N_sample]
    return adj, features, prop

class Progress(keras.callbacks.Callback):
    """
    A simple function to show the progress
    """
    def on_epoch_end(self, epoch, logs):
        if epoch % 1 == 0:
             print('epoch-----', epoch, logs)


#load data
adj, features, props = load_data()

#Define the model
for i, lr in enumerate([1e-3]):
    model = mymodel()
    #optimizer = tf.keras.optimizers.RMSprop(0.001)
    optimizer = tf.keras.optimizers.Adam(lr=lr, decay=1e-6)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    history = model.fit(x=[features, adj], y=props, batch_size=100, 
                        epochs=1000, validation_split=0.1, verbose=0,
                        callbacks=[Progress()]
                       )
    
    #Analyze the results
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.figure()
    plt.subplot(211)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label = 'Val Error')
    plt.legend()
    
    plt.subplot(212)
    res = model.predict([features,adj]).flatten()
    r2 = 'r2: {:.4f} '.format(r2_score(res, props))
    mae = 'mae: {:.3f}'.format(mean_absolute_error(res, props))
    plt.scatter(props, res, label=r2+mae)
    plt.legend()
    plt.xlabel('True values')
    plt.ylabel('Predictions')
    plt.savefig(str(i)+'.png')
    
    print(r2+mae)
    print(model.summary())

# Recreate the exact same model, including weights and optimizer.
#model.save('my_model.h5')
#new_model = keras.models.load_model('my_model.h5')
#new_model.summary()
