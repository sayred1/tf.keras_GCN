import tensorflow as tf

def skip_connection(_features, features):
    if(_features.shape[2] != features.shape[2]):
        out_dim = int(_features.shape[2])
        #going to have to figure out a way to change the arguments of tf.layers.dense s.t i can use tf.keras.dense()
        _features = tf.nn.relu(_features + tf.layers.dense(features, units = out_dim, use_bias = False)) # bad practice?
    else:
        _features = tf.nn.relu(_features+features)
    return(_features)

def GCN(features,adjacency,weights,bias):
    dim = int(weights.get_shape()[1])
    num_atoms = adjacency.shape[1]
    _bias = tf.reshape(tf.tile(bias, [num_atoms]), [num_atoms,dim])

    _features = tf.einsum('ijk,kl->ijl', features, weights) + _bias # makesure you can explain this
    _features = tf.matmul(adjacency,_features)
    _features = skip_connection(_features, features)
    return(tf.nn.relu(_features))
