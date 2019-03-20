import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error, r2_score
import logging
from tensorflow.keras import backend as K
from optparse import OptionParser
from time import time
K.set_floatx('float64')
logging.basicConfig(filename='test.log', level=logging.INFO)

class GCN(layers.Layer):
    """
    A graph convolution model
    """
    def __init__(self, output_dim, activation, **kwargs):
        self.output_dim = output_dim
        self.activation = activation
        super(GCN, self).__init__(**kwargs)

    def build(self, input_shape):
        #print(input_shape[0][2])
        shape = tf.TensorShape((input_shape[0][2], self.output_dim))
        shape = [int(shape[0]),int(shape[1])] # [50 , 32]

        self.kernel = self.add_weight(name='kernel',
                                      shape=shape,
                                      initializer='glorot_uniform',
                                      #initializer='he_uniform',
                                      trainable=True)
                                      #dtype=tf.float64)

        self.bias = self.add_weight(name='bias',
                                    shape=[shape[1]],
                                    initializer='glorot_uniform',
                                    #initializer='he_uniform',
                                    trainable=True)
                                    #dtype=tf.float64)

    def call(self, input):
        X, A = input[0], input[1]
        dim = self.kernel.get_shape()[1]
        num_atoms = A.get_shape()[1]
        _b = tf.reshape(tf.tile(self.bias, [num_atoms]), [num_atoms, dim])
        _X = tf.einsum('ijk,kl->ijl', X, self.kernel) + _b
        _X = tf.matmul(A, _X)
        _X = get_skip_connection(_X, X)
        return self.activation(_X)

class AGCN(layers.Layer):
    """
    An attention graph convolution model
    """
    def __init__(self, output_dim, activation, **kwargs):
        self.output_dim = output_dim
        self.activation = activation
        super(AGCN, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[0][2], self.output_dim))
        shape = [int(shape[0]),int(shape[1])] # [50 , 32]
       
        # att. weight, conv. weight, and conv. bias calculated in K = 4 subspaces
        self.conv_weight = []
        self.conv_bias = []
        self.attn_weight = [] 
        for i in range(4):
            attn_kernel = self.add_weight(name='attn-kernel'+str(i),
                                          shape=[shape[1],shape[1]],
                                          initializer='glorot_uniform',
                                          trainable = True)
            self.attn_weight.append(attn_kernel)

            kernel = self.add_weight(name='kernel',
                                          shape=shape,
                                          initializer='glorot_uniform',
                                          trainable=True)
            self.conv_weight.append(kernel)

            bias = self.add_weight(name='bias',
                                        shape=[shape[1]],
                                        initializer='glorot_uniform',
                                        trainable=True)
            self.conv_bias.append(bias)

    def call(self, input):
        X, A = input[0], input[1]
        dim = self.conv_weight[0].get_shape()[1]
        num_atoms = A.get_shape()[1]
        X_total = [] 
        A_total = []
        for i in range(len(self.conv_weight)):                  # _X and A (attention coeff.) calculed in K = 4 subspaces  
            _b = tf.reshape(tf.tile(self.conv_bias[i], [num_atoms]), [num_atoms, dim])
            _h = tf.einsum('ijk,kl->ijl', X, self.conv_weight[i]) + _b
            _A = attn_matrix(A, _h, self.attn_weight[i])        # calculates the attention coefficient (changes weights between atoms)
            _h = tf.nn.relu(tf.matmul(_A, _h))
            X_total.append(_h)
            A_total.append(_A)
        _X = tf.nn.relu(tf.reduce_mean(X_total, 0))             # activation of the mean of K = 4 updated feature calculations
        _A = tf.reduce_mean(A_total, 0)                         # mean of K = 4 attention coefficients
        _X = get_skip_connection(_X, X)                         # basic skip-connection
        return self.activation(_X)

class AGGCN(layers.Layer):
    """
    An attention gated graph convolution model
    """
    def __init__(self, output_dim, activation, **kwargs):
        self.output_dim = output_dim
        self.activation = activation
        super(AGGCN, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[0][2], self.output_dim))
        shape = [int(shape[0]),int(shape[1])] # [50 , 32]
       
        # att. weight, conv. weight, and conv. bias calculated in K = 4 subspaces
        self.conv_weight = []
        self.conv_bias = []
        self.attn_weight = [] 
        for i in range(4):
            attn_kernel = self.add_weight(name='attn-kernel'+str(i),
                                          shape=[shape[1],shape[1]],
                                          initializer='glorot_uniform',
                                          trainable = True)
            self.attn_weight.append(attn_kernel)

            kernel = self.add_weight(name='kernel',
                                          shape=shape,
                                          initializer='glorot_uniform',
                                          trainable=True)
            self.conv_weight.append(kernel)

            bias = self.add_weight(name='bias',
                                        shape=[shape[1]],
                                        initializer='glorot_uniform',
                                        trainable=True)
            self.conv_bias.append(bias)
        self.gate_bias = self.add_weight(name = 'gate_bias',
                                         shape = [shape[1]],
                                         initializer='glorot_uniform',
                                         dtype=tf.float64)

    def call(self, input):
        X, A = input[0], input[1]
        dim = self.conv_weight[0].get_shape()[1]
        num_atoms = A.get_shape()[1]
        X_total = [] 
        A_total = []
        for i in range(len(self.conv_weight)):                  # _X and A (attention coeff.) calculed in K = 4 subspaces  
            _b = tf.reshape(tf.tile(self.conv_bias[i], [num_atoms]), [num_atoms, dim])
            _h = tf.einsum('ijk,kl->ijl', X, self.conv_weight[i]) + _b
            _A = attn_matrix(A, _h, self.attn_weight[i])        # calculates the attention coefficient (changes weights between atoms)
            _h = tf.nn.relu(tf.matmul(_A, _h))
            X_total.append(_h)
            A_total.append(_A)
        _X = tf.nn.relu(tf.reduce_mean(X_total, 0))             # activation of the mean of K = 4 updated feature calculations
        _A = tf.reduce_mean(A_total, 0)                         # mean of K = 4 attention coefficients
        _X = self.activation(_X)
        if( int(X.get_shape()[2]) != dim ):
            X = tf.layers.dense(X, dim, use_bias=False)
        coeff = get_gate_coeff(_X,X,dim,self.gate_bias)     # gate coefficient (determines if _X is updated from recent layers)
        _X = tf.multiply(_X,coeff) + tf.multiply(X,1.0-coeff)
        return self.activation(_X)


class GGCN(layers.Layer):
    """
    A gated graph convolution model
    """
    def __init__(self, output_dim, activation, **kwargs):
        self.output_dim = output_dim
        self.activation = activation
        super(GGCN, self).__init__(**kwargs)

    def build(self, input_shape):
        #print(input_shape[0][2])
        shape = tf.TensorShape((input_shape[0][2], self.output_dim))
        shape = [int(shape[0]),int(shape[1])] # [50 , 32]

        self.kernel = self.add_weight(name='kernel',
                                      shape=shape,
                                      initializer='glorot_uniform',
                                      trainable=True)

        self.bias = self.add_weight(name='bias',
                                    shape=[shape[1]],
                                    initializer='glorot_uniform',
                                    trainable=True)
        self.gate_bias = self.add_weight(name = 'gate_bias',
                                         shape = [shape[1]], 
                                         initializer='glorot_uniform',
                                         trainable=True)
    def call(self, input):
        X, A = input[0], input[1]
        dim = self.kernel.get_shape()[1]
        num_atoms = A.get_shape()[1]
        _b = tf.reshape(tf.tile(self.bias, [num_atoms]), [num_atoms, dim])
        _X = tf.einsum('ijk,kl->ijl', X, self.kernel) + _b
        _X = tf.matmul(A, _X)
        _X = self.activation(_X)
        if( int(X.get_shape()[2]) != dim ):
            X = tf.layers.dense(X, dim, use_bias=False)
        coeff = get_gate_coeff(_X,X,dim,self.gate_bias)     # gate coefficient (determines if _X is updated from recent layers)
        _X = tf.multiply(_X,coeff) + tf.multiply(X,1.0-coeff)
        return self.activation(_X)

class G2N(layers.Layer):
    """
    A layer to sum the node feature to preserve the permutation invariance
    """
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(G2N, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[2], self.output_dim))
        shape = [int(shape[0]),int(shape[1])] # [50 , 32]

        self.kernel = self.add_weight(name='kernel',
                                      shape=shape,
                                      #initializer='he_uniform',
                                      initializer='glorot_uniform',
                                      trainable=True)
                                      #dtype=tf.float64)
        self.bias = self.add_weight(name='bias',
                                    shape=[shape[1]],
                                    #initializer='he_uniform',
                                    initializer='glorot_uniform',
                                    trainable=True)
                                    #dtype=tf.float64)


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
        for i, hid in enumerate(graph_layers):
            if model_name == 'GCN':
                self.graph_layers.append(GCN(hid, name='GCN'+str(i), activation=tf.nn.relu)) 
            elif model_name == 'GGCN':
                self.graph_layers.append(GGCN(hid, name='GGCN'+str(i), activation=tf.nn.relu)) 
            elif model_name == 'AGCN':
                self.graph_layers.append(AGCN(hid, name='AGCN'+str(i), activation=tf.nn.relu)) 
            elif model_name == 'AGGCN':
                self.graph_layers.append(AGCN(hid, name='AGGCN'+str(i), activation=tf.nn.relu)) 

        self.g2n = G2N(graph_layers[-1], name='Readout')
        for i, mlp in enumerate(mlp_layers):
            self.mlp_layers.append(layers.Dense(mlp, name='MLP'+str(i), activation=tf.nn.relu))
        self.mlp_layers.append(layers.Dense(1, name='MLP'+str(i+1) ))

    def call(self, input):
        x, A = input[0], input[1]
        for model in self.graph_layers:
            x = model([x, A])
        x = self.g2n(x)
        for model in self.mlp_layers:
            x = model(x)
        return x

def get_gate_coeff(X1, X2, dim, _b):
    num_atoms = int(X1.get_shape()[1])
    _b = tf.reshape(tf.tile(_b, [num_atoms]), [num_atoms, dim])
    X1 = tf.layers.dense(X1, units=dim, use_bias=False)
    X2 = tf.layers.dense(X2, units=dim, use_bias=False)
    output = tf.nn.sigmoid(X1+X2+_b)

    return output

def get_skip_connection(_X, X):
    if( int(_X.get_shape()[2]) != int(X.get_shape()[2]) ):
       out_dim = int(_X.get_shape()[2])
       _X = _X + tf.layers.dense(X, units = out_dim, use_bias=False)
    else:
       _X = _X + X 

    return _X 

def attn_matrix(A, X, attn_weight):
    # A : [batch, N, N]
    # X : [batch, N, F']
    # weight_attn : F'
    num_atoms = int(X.get_shape()[1])
    hidden_dim = int(X.get_shape()[2])

    _X1 = tf.einsum('ij,ajk->aik', attn_weight, tf.transpose(X, [0,2,1]))
    _X2 = tf.matmul(X, _X1)
    _A = tf.multiply(A, _X2)
    _A = tf.nn.tanh(_A)

    return _A

def load_data(ids, unit=10000):
    """
    load the source data from somewhere
    """
    features = np.load('../../augmented-GCN/database/ZINC/features/'+str(ids[0])+'.npy')
    adj = np.load('../../augmented-GCN/database/ZINC/adj/'+str(ids[0])+'.npy')
    
    for i in range(ids[0]+1, ids[1]):
        fea0 = np.load('../../augmented-GCN/database/ZINC/features/'+str(i)+'.npy')
        adj0 = np.load('../../augmented-GCN/database/ZINC/adj/'+str(i)+'.npy')
        features = np.concatenate((features, fea0), axis=0) 
        adj = np.concatenate((adj, adj0), axis=0)
    N_sample = len(features)
    prop = np.load('../../augmented-GCN/database/ZINC/logP.npy')[ids[0]*unit:ids[1]*unit]
    prop = np.reshape(prop, [len(prop), 1])
    return adj, features, prop

class Progress(keras.callbacks.Callback):
    """
    A simple function to show the progress
    """
    def on_epoch_end(self, epoch, logs):
        if epoch % 1 == 0:
             print('epoch-----', epoch, logs)


if __name__ == '__main__':
    # ------------------------ Options -------------------------------------
    parser = OptionParser()
    parser.add_option("-m", "--model", dest="model", default='GCN',
                      help="training model, GCN(default)")
    parser.add_option("-g", "--graph", dest="graph", default='32,32,32', #nargs='+', type=int,
                      help="graph_layers in list, 32, 32, 32(default)")
    parser.add_option("-l", "--mlp", dest="mlp", default='64,64', #nargs='+', type=int, 
                      help="mlp layers in list, 64,64(default)")
    parser.add_option("-e", "--epoch", dest="epoch", default=100, type=int,
                      help="number of epochs, 200 (default)")
    (options, args) = parser.parse_args()

    # Define the adjustable parameters
    train, test = [0, 40], [40, 50]
    #train, test = [0, 2], [2, 3]
    model_name = options.model
    graph_layers = [int(item) for item in options.graph.split(',')]
    mlp_layers = [int(item) for item in options.mlp.split(',')]
    epoch = options.epoch
    lr = 1e-3
    
    # Load data
    adj1, features1, props1 = load_data(train)
    adj2, features2, props2 = load_data(test)
    
    # Define the model
    title = model_name+'-epoch'+str(epoch)
    title += '+Graph' 
    for i, lay in enumerate(graph_layers):
        title += str(lay)
        if i+1 < len(graph_layers):
            title += '-'
    title += '+MLP'
    for i, lay in enumerate(mlp_layers):
        title += str(lay)
        if i+1 < len(mlp_layers):
            title += '-'
    logging.info('The input model is {:s}'.format(title))
    time0 = time()
    model = mymodel(model_name, graph_layers, mlp_layers)

    #optimizer = tf.keras.optimizers.RMSprop(0.001)
    optimizer = tf.keras.optimizers.Adam(lr=lr, decay=1e-6)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error'])

    # callback function to save the results to tensorboard
    #tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', 
    #                                         histogram_freq=10, 
    #                                         write_graph=True, 
    #                                         write_images=True)

    history = model.fit(x=[features1, adj1], y=props1, batch_size=64, 
                        epochs=epoch, verbose=0,
                        callbacks=[Progress(), keras.callbacks.CSVLogger('training.log')]
                       )
    logging.info('Training is completed in {:.2f} minutes'.format((time()-time0)/60))
    logging.info('Total number of parameters: {:d}'.format(history.model.count_params()))
    
    # Analyze the results
    res1 = model.predict([features1,adj1])
    res2 = model.predict([features2,adj2])
    print(res1.shape, props1.shape)
    print(r2_score(res1, props1), mean_absolute_error(res1, props1), keras.losses.mae(res1, props1))
    #print(model.evaluate(x=[features1, adj1], y=props1, batch_size=100))
    logging.info('Train in {:d} samples, r2 {:.4f} mae {:.4f}'.format(len(res1), r2_score(res1, props1), mean_absolute_error(res1, props1)))
    logging.info('Test  in {:d} samples, r2 {:.4f} mae {:.4f}'.format(len(res2), r2_score(res2, props2), mean_absolute_error(res2, props2)))
    
    #history = model.fit(x=[features1, adj1], y=props1, batch_size=64, 
    #                    epochs=2, verbose=0,
    #                    callbacks=[Progress(), keras.callbacks.CSVLogger('training.log')]
    #                   )

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.figure(figsize=(10, 8))
    plt.subplot(211)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.legend()
    
    plt.subplot(212)
    
    label1 = 'Train: r2: {:.4f} mae: {:.4f} in {:d}'.format(r2_score(res1, props1), 
                                             mean_absolute_error(res1, props1),
                                                                  len(props1))
    label2 = 'test:  r2: {:.4f} mae: {:.4f} in {:d}'.format(r2_score(res2, props2), 
                                             mean_absolute_error(res2, props2),
                                                                  len(props2))
    plt.scatter(props1, res1, alpha=0.6, label=label1)
    plt.scatter(props2, res2, alpha=0.6, label=label2)
    plt.legend()
    plt.xlabel('True values')
    plt.ylabel('Predictions')
    plt.savefig(title+'.png')
    np.savetxt(title+'.txt', hist['mean_absolute_error'])
    print(history.model.summary())
    
    # Recreate the exact same model, including weights and optimizer.
    #model.save('my_model.h5')
    #new_model = keras.models.load_model('my_model.h5')
    #new_model.summary()
