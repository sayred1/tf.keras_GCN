import tensorflow as tf

def skip_connection(_features, features):
    if(_features.shape[2] != features.shape[2]):
        out_dim = int(_features.get_shape()[2])
        _features = tf.nn.relu(_features + tf.layers.dense(features, units = out_dim, use_bias = False)) # bad practice?
    else:
        _features = tf.nn.relu(_X+X)
    return(_features)

def GCN(features,adjacency,weights,bias):
    dim = weights.shape[1]
    num_atoms = int(features.get_shape()[1])
    _bias = tf.reshape(tf.tile(bias, [num_atoms]), [num_atoms,dim])

    _features = tf.einsum('ijk, kl -> ijl', features, weights) + _bias # makesure you can explain this
    _features = tf.matmul(adjacency,_features)

    _features = skip_connection(_features, features)
    print('the shape', _features)
    return(_features)
