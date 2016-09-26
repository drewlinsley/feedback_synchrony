from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.models.rnn.ptb import reader
from utils import * #_variable_with_weight_decay
import prepare_mnist_data
import data_loader
import os

#Load data
which_data = 'cluttered_mnist'
X_train_raw,y_train_temp,X_test_raw,y_test_temp,train_num,im_size = data_loader.train(which_data)
X_train_raw = X_train_raw.astype('float32')
data_mu = np.mean(X_train_raw)
data_std = np.std(X_train_raw)
X_train_raw-=data_mu
X_train_raw/=data_std

# Model hyperparameters/training parameters
settings = {'batch_size':30,
'num_steps':5,
'epochs':30,
'num_afferents':1, #set to some # > # of layers for fully connected
'filters':[40,40,40],
'filter_r':[3,3,3],
'filter_w':[3,3,3],
'padding':[0,0,0],
'stride':[1,1,1],
'pool_size':2,
'output_shape':1, #regression
'la':0.1, #l2 regularization for FC layer
'dropout_prob':.5,
'channels':1,
'ckpt_dir':'./ckpt_dir',
'model_name':'complex', #real or complex
'gpu_number':0,
'restore_model':False,
'height':im_size[0],
'width':im_size[1],
'FC_dim':(height // pool_size) **2 * filters[-1],
'init' = tf.contrib.layers.xavier_initializer_conv2d, #Weight inits #tf.contrib.layers.xavier_initializer
'inner_init' = orthogonal_initializer,
'activation' = orthogonal_initializer,
'inner_activation' = orthogonal_initializer,
'activation'= tf.tanh,
'inner_activation'= tf.nn.relu6, #hard sigmoid
}

# Build Model
session, merged, saver, optim, writer = model.build_model(settings)
session.run(tf.initialize_all_variables())
if restore_model == True:
  ckpt = tf.train.get_checkpoint_state(ckpt_dir)
  saver.restore(session,ckpt.model_checkpoint_path)

#Train model
session, result = model.batch_train(session, merged, saver, optim, writer, X_train_raw, y_train_temp, settings)