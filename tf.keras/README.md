# TF-Keras implementation of GCN

## Run
```
(tf_gpu) tf.keras$ python tf-keras.py -h
Usage: tf-keras.py [options]

Options:
  -h, --help            show this help message and exit
  -m MODEL, --model=MODEL
                        training model, GCN(default)
  -g GRAPH, --graph=GRAPH
                        graph_layers in list, [32, 32, 32](default)
  -l MLP, --mlp=MLP     mlp layers in list, [128, 128](default)
  -e EPOCH, --epoch=EPOCH
                        number of epochs, 100 (default)
```


## Tensorboard 
It is generally good to visuallize the results via powerful tensorboard. To do so, one can just use the following callback function to save the results to tensorboard in `model.fit()`
```python
    tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', 
                                             histogram_freq=10, 
                                             write_graph=True, 
                                             write_images=True)
```

All of the log directories are contained in the summaries directory. They're are broken up into four different directories 
which contain the training process with four different optimizers and initial learning rate 0.001.

To access them with tensorboard, type in:
  'tensorboard --logdir=summaries/' in the terminal where summaries exists.

A message SIMILAR to this will pop up:
  'TensorBoard __version__ at http://link-to-url:####'

Don't copy the whole link, take the port '####' and type into the browser: 
  'localhost:####'. This port number usually defaults to 6006.

If the session number #### is not available, specify another port less than 65535 like this:
  'tensorboard --logdir=summaries --port=65533'.

When you see the message, copy the session number and type into the browser: 
  'localhost:65533'
