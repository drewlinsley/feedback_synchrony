from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.models.rnn.ptb import reader
import os
from ops import data_loader, prepare_mnist_data 
from ops import model
#from ops import test_model as model
from ops.utils import *
import time

#Set output directories
meta_output = './meta_output'
if not os.path.exists(meta_output):
    os.makedirs(meta_output)

#Load data
which_data = 'coco'#'multi_mnist'
X_train_raw,y_train_temp,X_test_raw,y_test_temp,train_num,im_size,num_channels,cats = data_loader.train(which_data)
X_train_raw,data_mu,data_std = data_loader.normalize(X_train_raw,zm=True,uv=True)

# Model hyperparameters/training parameters
pool_size = 2#[2,2,2]#2 
filters = [60,60,60]#[128,64,32]#[60,60,60]
settings = {'batch_size':30,
'num_steps':5,
'epochs':30,
'num_afferents':1, #set to some # > # of layers for fully connected
'filters':filters,
'filter_r':[7,3,3],
'filter_w':[7,3,3],
'padding':[0,0,0],
'stride':[1,1,1],
'pool_size':pool_size,
'output_shape':np.max(cats),#len(np.unique(cats)), #regression or classification (multiple vals)
'la':1, #l2 regularization for FC layer
'dropout_prob':.2,
'channels':num_channels,
'ckpt_dir':'./ckpt_dir',
'model_name':'complex',#'no_sync_complex', #real or complex
'extra_tag':'_' + which_data + '_' + time.strftime('%H:%M:%S'),
'gpu_number':0, #keep at 0 unless using multi-gpu
'restore_model':True,
'height':im_size[0],
'width':im_size[1],
'num_fc':1,
'init':tf.contrib.layers.xavier_initializer_conv2d, #Weight inits #tf.contrib.layers.xavier_initializer
'inner_init':orthogonal_initializer,
'activation':orthogonal_initializer,
'inner_activation':orthogonal_initializer,
'activation':tf.tanh,
'inner_activation':tf.nn.relu6, #hard sigmoid
'which_data':which_data
}
s = Map(settings)

# Build Model
session, init_vars, merged, saver, optim, writer, cost, keep_prob, X, targets, Wc, Uc, Wg, Ug, pred, accuracy = model.build_model(s)
session.run(init_vars)
if s.restore_model == True:
  ckpt = tf.train.get_checkpoint_state(ckpt_dir)
  saver.restore(session,ckpt.model_checkpoint_path)

import ipdb;
ipdb.set_trace()
#Train model
session, result = model.batch_train(session, merged, saver, optim, writer, cost, keep_prob, accuracy, X, targets, X_train_raw, y_train_temp, X_test_raw, y_test_temp, s)

#Save settings for later
np.savez(meta_output + '/' + s.model_name + s.extra_tag,settings=settings)
