from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.models.rnn.ptb import reader
from utils import * #_variable_with_weight_decay
import prepare_mnist_data
import data_loader

#Load data
which_data = 'cluttered_mnist'
X_train_raw,y_train_temp,X_test_raw,y_test_temp,train_num,im_size = data_loader.train(which_data)
X_train_raw = X_train_raw.astype('float32')
data_mu = np.mean(X_train_raw)
data_std = np.std(X_train_raw)
X_train_raw-=data_mu
X_train_raw/=data_std

# Hyperparameters
batch_size = 10
num_steps = 5
epochs = 20
num_afferents = 1
filters = [80,80,80]
filter_r = [3,3,3]
filter_w = [3,3,3]
padding = [0,0,0]
stride = [1,1,1]
pool_size = 2
output_shape = 1 #regression
la = 0.01 #l2 regularization for FC layer
channels = 1
height = im_size[0]
width = im_size[1]
FC_dim = (height // pool_size) **2 * filters[-1]

#Weight inits
#init = tf.contrib.layers.xavier_initializer
init = tf.contrib.layers.xavier_initializer_conv2d
inner_init = orthogonal_initializer
activation = orthogonal_initializer
inner_activation = orthogonal_initializer
activation= tf.tanh
inner_activation= tf.nn.relu6 #hard sigmoid

#Initialize some variables
prev_channels = channels
prev_height = height
Wi =[]; Ui = []; Wib = []; Uib = []; 
Wf =[]; Uf = []; Wfb = []; Ufb = []; 
Wo =[]; Uo = []; Wob = []; Uob = []; 
Wc =[]; Uc = []; Wcb = []; Ucb = []; 
Wg =[]; Ug = []; Wgb =[]; Ugb = []; 
Sc = []; Sh = []; state = [];

hh = [height]
hc = [prev_channels]
## Build Model
with tf.device('/gpu:0'):
  lr = tf.placeholder(tf.float32, [])
  X = tf.placeholder(tf.float32, [batch_size, num_steps, height, width, channels]) #batch,time,height,width,channels
  targets = tf.placeholder(tf.float32, [batch_size]) #replace num_steps with 1 if doing a single prediction

  #FC 
  fc1_weights = tf.Variable(  # fully connected, depth 512.
      tf.truncated_normal([FC_dim, output_shape],
                          stddev=0.1))
  fc1_biases = tf.Variable(tf.constant(0.1, shape=[1]))

  #conv lstm
  for i in range(0,len(filters)):
    in_size_w = [filter_r[i], filter_w[i], prev_channels, filters[i]]
    in_size_u = [filter_r[i], filter_w[i], filters[i],filters[i]]

    # I
    Wi.append(tf.get_variable(name="Wi_%d" % i, shape=in_size_w,initializer=init()))
    Ui.append(tf.get_variable(name="Ui_%d" % i, shape=in_size_u,initializer=inner_init()))
    Wib.append(tf.Variable(tf.constant(0.0, shape=[filters[i]],dtype=tf.float32),trainable=True,name="Wib_%d" % i))
    Uib.append(tf.Variable(tf.constant(0.0, shape=[filters[i]],dtype=tf.float32),trainable=True,name="Uib_%d" % i))
    # F
    Wf.append(tf.get_variable(name="Wf_%d" % i, shape=in_size_w,initializer=init()))
    Uf.append(tf.get_variable(name="Uf_%d" % i, shape=in_size_u,initializer=inner_init()))
    Wfb.append(tf.Variable(tf.constant(0.0, shape=[filters[i]],dtype=tf.float32),trainable=True,name="Wfb_%d" % i))
    Ufb.append(tf.Variable(tf.constant(0.0, shape=[filters[i]],dtype=tf.float32),trainable=True,name="Ufb_%d" % i))
    # C
    Wc.append(tf.get_variable(name="Wc_%d" % i, shape=in_size_w,initializer=init()))
    Uc.append(tf.get_variable(name="Uc_%d" % i, shape=in_size_u,initializer=inner_init()))
    Wcb.append(tf.Variable(tf.constant(0.0, shape=[filters[i]],dtype=tf.float32),trainable=True,name="Wcb_%d" % i))
    Ucb.append(tf.Variable(tf.constant(0.0, shape=[filters[i]],dtype=tf.float32),trainable=True,name="Ucb_%d" % i))
    # O
    Wo.append(tf.get_variable(name="Wo_%d" % i, shape=in_size_w,initializer=init()))
    Uo.append(tf.get_variable(name="Uo_%d" % i, shape=in_size_u,initializer=inner_init()))
    Wob.append(tf.Variable(tf.constant(0.0, shape=[filters[i]],dtype=tf.float32),trainable=True,name="Wob_%d" % i))
    Uob.append(tf.Variable(tf.constant(0.0, shape=[filters[i]],dtype=tf.float32),trainable=True,name="Uob_%d" % i))
    # Gated feedback (Bengio group, 2015)
    Wg.append(tf.get_variable(name="Wg_%d" % i, shape=in_size_w, initializer=init())) #vector with current filter size... scalar weight on activation maps need to reshape into a tensor
    Ug.append(tf.get_variable(name="Ug_%d" % i, shape=[filter_r[i], filter_w[i], np.sum(filters),np.sum(filters)],initializer=inner_init())) #vector with num_filters size... scalar weight on activation maps need to reshape into a tensor.... accepts all layers... fix this
    Wgb.append(tf.Variable(tf.constant(0.0, shape=[filters[i]],dtype=tf.float32),trainable=True,name="Wgb_%d" % i))
    Ugb.append(tf.Variable(tf.constant(0.0, shape=[np.sum(filters)],dtype=tf.float32),trainable=True,name="Ugb_%d" % i))
    prev_channels = filters[i]
    # Add synchrony here
    Sc.append(tf.Variable(tf.constant(0.5, shape=[2],dtype=tf.float32),trainable=True,name="Sc_%d" % i))
    Sh.append(tf.Variable(tf.constant(0.5, shape=[2],dtype=tf.float32),trainable=True,name="Sh_%d" % i))
    # Preallocate the state
    state.append(tf.zeros([batch_size,prev_height,prev_height,prev_channels])) #Consider concatenating Weight matrices... Should be much faster
    #prev_height = (prev_height + 2 * padding[i] - filter_r[i]) / stride[i] + 1
    prev_height = height
    prev_channels = channels
    hh.append(prev_height)
    hc.append(prev_channels)
  # For the output
  output_weights = tf.get_variable(name="output_weights", shape=[filters[-1], output_shape], initializer=inner_init())
  output_bias = tf.Variable(tf.zeros([output_shape]), name="output_bias")

  #Start the RNN loop
  num_hidden_layers = len(filters)
  c = state[0]
  prev_concat_h = tf.zeros([batch_size,prev_height,prev_height,np.sum(filters)])#for now all filters have to be the same size... in the future use up/downsampling
  loss = tf.zeros([])
  # TODO: prev concat h
  for time_step in range(num_steps):
    h_prev = X[:, time_step, :, :, :]
    for layer in range(num_hidden_layers):
      layer_strides = np.repeat(stride[layer],4).tolist()
      #Facing the input
      xi = tf.nn.conv2d(h_prev, Wi[layer] * Wib[layer], layer_strides, padding='SAME')
      xf = tf.nn.conv2d(h_prev, Wf[layer] * Wfb[layer], layer_strides, padding='SAME')
      xo = tf.nn.conv2d(h_prev, Wo[layer] * Wob[layer], layer_strides, padding='SAME')
      xc = tf.nn.conv2d(h_prev, Wc[layer] * Wcb[layer], layer_strides, padding='SAME')
      #Facing the hidden layer
      hi = tf.nn.conv2d(state[layer], Ui[layer] * Uib[layer], layer_strides, padding='SAME')
      hf = tf.nn.conv2d(state[layer], Uf[layer] * Ufb[layer], layer_strides, padding='SAME')
      ho = tf.nn.conv2d(state[layer], Uo[layer] * Uob[layer], layer_strides, padding='SAME')
      hc = tf.nn.conv2d(state[layer], Uc[layer] * Ucb[layer], layer_strides, padding='SAME')
      #Fix complex gates after applying activations
      i = (inner_activation(xi + hi)) #need to implement the complex-valued ops fix_complex_gates
      f = (inner_activation(xf + hf))
      o = (inner_activation(xo + ho)) #The original implementation handled the h's seperately

      # Main contribution of paper:
      #con_range = range(layer + np.min([layer + num_afferents, num_hidden_layers])) #need to adjust the hidden state concat
      con_range = range(num_hidden_layers)
      #Gates  applied to every afferent
      gates = [inner_activation(tf.nn.conv2d(h_prev, Wg[idx] * Wgb[idx], layer_strides, padding='SAME') + 
        tf.reduce_sum(tf.reshape(tf.nn.conv2d(prev_concat_h, Ug[idx] * Ugb[idx], layer_strides, padding='SAME'),(batch_size,prev_height,prev_height,filters[idx],num_hidden_layers)),4)) for idx in con_range] #restricted to num_afferents levels above the current... is this conv or element


      #Gated_prev_timestep is the activations from all afferents weighted by Gates
      gated_prev_timestep = [gates[idx] * tf.nn.conv2d(state[layer], Uc[idx] * Ucb[idx], layer_strides, padding='SAME') for idx in con_range]
      #c is now calculated as tanh(current hidden content + sum of the gated afferents)
      new_c = activation(tf.nn.conv2d(h_prev, Wc[layer] * Wcb[layer], layer_strides, padding='SAME') + tf.add_n(gated_prev_timestep))

      #Get new h and c as per usual
      c = tf.mul(f, c) + tf.mul(i, new_c) #complex multiplication here
      state[layer] = tf.mul(o, activation(c))

    #pool_state = tf.nn.max_pool(state[layer],ksize=[1,pool_size,pool_size,1],strides=[1,pool_size,pool_size,1],padding='VALID',name='end_pool')
    #reshape then fc... maybe a 1d lstm here...
    #if regression

    #cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
    #if classsificaiton
    #...
    #res_pool_state = tf.reshape(pool_state,[batch_size,prev_height//pool_size*prev_height//pool_size*filters[-1]])
    #logits = tf.nn.bias_add(tf.matmul(state[num_hidden_layers-1], output_weights), output_bias)
    #step_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets[:, time_step])
    #regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases))
    #loss += 5e-4 + regularizers #tf.reduce_sum(step_loss)
    prev_concat_h = tf.concat(3, state)
  pool_state = tf.nn.max_pool(state[layer],ksize=[1,pool_size,pool_size,1],strides=[1,pool_size,pool_size,1],padding='VALID',name='end_pool')
  res_pool_state = tf.reshape(pool_state,[batch_size,prev_height//pool_size*prev_height//pool_size*filters[-1]])
  pred = tf.add(tf.matmul(res_pool_state,fc1_weights),fc1_biases)
  regularizers = tf.add(tf.nn.l2_loss(fc1_weights),tf.nn.l2_loss(fc1_biases))
  error_loss = tf.reduce_sum((tf.pow(pred-targets, 2))/ batch_size)
  reg_loss = la * regularizers
  cost = error_loss + reg_loss  

tf.scalar_summary("cost", cost)
merged = tf.merge_all_summaries()
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#config.log_device_placement=True
#session = tf.Session(config)
session = tf.Session()
writer = tf.train.SummaryWriter("summaries/gfrnn", session.graph)

# Train Model
#optim = tf.train.GradientDescentOptimizer(lr).minimize(cost)
optim = tf.train.AdamOptimizer(1e-4).minimize(cost)
session.run(tf.initialize_all_variables())

costs = 0.0
iters = 0

data_size = X_train_raw.shape[0]
cv_folds = data_size // batch_size
for i in range(epochs):
  print('Epoch', i)
  x,y = prepare_mnist_data.adding_task(X_train_raw,y_train_temp,num_steps,data_size,im_size,'regress')
  cv_ind = range(data_size)
  np.random.shuffle(cv_ind)
  cv_ind = np.reshape(cv_ind,[cv_folds,batch_size])
  for idx in range(cv_folds):
    train_idx = cv_ind[idx,:]
    bx = x[train_idx,:,:,:,:]
    by = y[train_idx]
    result, step_cost, _, = session.run([merged, cost, optim],
                           #{X: x, targets: y, lr: 1.0 / (i + 1)})
                           {X: bx, targets: by})
    costs += step_cost
    iters += num_steps
    if iters % 10000 == 0:
      print(iters, np.exp(costs / iters))
      writer.add_summary(result, iters)
      writer.flush()

