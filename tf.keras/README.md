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
By executing the command of `python tf-keras.py`, one expects to get the following output
```
(tf_gpu) qzhu@cms:/scratch/qzhu/github/tf.keras_GCN/tf.keras$ python tf-keras.py   
2019-03-17 00:47:29.463025: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2019-03-17 00:47:29.690409: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties: 
name: GeForce GTX 1070 major: 6 minor: 1 memoryClockRate(GHz): 1.835
pciBusID: 0000:02:00.0
totalMemory: 7.93GiB freeMemory: 7.78GiB
2019-03-17 00:47:29.690465: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2019-03-17 00:47:29.974798: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-03-17 00:47:29.974828: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0 
2019-03-17 00:47:29.974834: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N 
2019-03-17 00:47:29.975024: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 7507 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1070, pci bus id: 0000:02:00.0, compute capability: 6.1)
epoch----- 0 {'loss': 2.398301799109143, 'mean_absolute_error': 1.2169866135756104}
epoch----- 1 {'loss': 1.6976235902712051, 'mean_absolute_error': 1.02939776994836}
epoch----- 2 {'loss': 1.1219212926862308, 'mean_absolute_error': 0.8272949244816248}
epoch----- 3 {'loss': 0.7887655994379076, 'mean_absolute_error': 0.6952147411245937}
epoch----- 4 {'loss': 0.6592997922526347, 'mean_absolute_error': 0.6361560752120833}
epoch----- 5 {'loss': 0.5235356713893753, 'mean_absolute_error': 0.5600877813741073}
epoch----- 6 {'loss': 0.44730538090674593, 'mean_absolute_error': 0.5196788608133286}
epoch----- 7 {'loss': 0.40022315918984136, 'mean_absolute_error': 0.4903589731418454}
epoch----- 8 {'loss': 0.38994625816791045, 'mean_absolute_error': 0.4822593380455136}
epoch----- 9 {'loss': 0.3316976324864056, 'mean_absolute_error': 0.443656754307256}
epoch----- 10 {'loss': 0.3068501940933695, 'mean_absolute_error': 0.4275905880915735}
epoch----- 11 {'loss': 0.29347837010943456, 'mean_absolute_error': 0.4196605755370023}
epoch----- 12 {'loss': 0.2857240041125045, 'mean_absolute_error': 0.4142143365290375}
epoch----- 13 {'loss': 0.2571666185800886, 'mean_absolute_error': 0.38921170224141727}
epoch----- 14 {'loss': 0.2473556796131501, 'mean_absolute_error': 0.3815712330816362}
epoch----- 15 {'loss': 0.27373683768095763, 'mean_absolute_error': 0.4057052894349517}
epoch----- 16 {'loss': 0.2615708643126004, 'mean_absolute_error': 0.39466690734576876}
epoch----- 17 {'loss': 0.20435170816012782, 'mean_absolute_error': 0.34595186365983244}
epoch----- 18 {'loss': 0.2050733167416051, 'mean_absolute_error': 0.3498082928655413}
....
....

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
GCN0 (GCN)                   multiple                  1888      
_________________________________________________________________
GCN1 (GCN)                   multiple                  1056      
_________________________________________________________________
GCN2 (GCN)                   multiple                  1056      
_________________________________________________________________
MLP0 (Dense)                 multiple                  4224      
_________________________________________________________________
MLP1 (Dense)                 multiple                  16512     
_________________________________________________________________
MLP2 (Dense)                 multiple                  129       
_________________________________________________________________
Readout (G2N)                multiple                  1056      
=================================================================
Total params: 25,921
Trainable params: 25,921
Non-trainable params: 0
```
and a `test.log` file saving the most important results:
```
INFO:root:The input model is GCN-epoch100+Graph32-32-32+MLP128-128
INFO:root:Training is completed in 1.770151 minutes
INFO:root:Train in 10000 samples, r2 0.9678 mae 0.1903
INFO:root:Test  in 10000 samples, r2 0.9558 mae 0.2123
```
and a figure showing the evolution/performance of training 
![plots](https://github.com/sayred1/tf.keras_GCN/blob/master/tf.keras/imgs/GCN-epoch100%2BGraph32-32-32%2BMLP128-128.png)

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
