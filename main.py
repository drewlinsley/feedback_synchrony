from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.models.rnn.ptb import reader
import os
from ops import data_loader, prepare_mnist_data, model
#from ops import atrous_model as model
from ops.utils import *
import time
from glob import glob

#Set output directories
meta_output = './meta_output'
if not os.path.exists(meta_output):
    os.makedirs(meta_output)

#Load data
which_data = 'cluttered_mnist_classification'#'cluttered_mnist_classification'##'multi_mnist'#coco
num_steps = 5
X_train_raw,y_train_temp,X_test_raw,y_test_temp,_,im_size,num_channels,cats = data_loader.train(which_data=which_data,num_steps=num_steps)
X_train_raw,data_mu,data_std = data_loader.normalize(X_train_raw,zm=True,uv=True)

# Model hyperparameters/training parameters
pool_size = 2#[2,2,2]#2 
filters = [60,60,60]#[128,64,32]#[60,60,60]
settings = {'batch_size':30,
'num_steps':num_steps,
'epochs':30,
'num_afferents':1, #set to some # > # of layers for fully connected
'filters':filters,
'filter_r':[7,3,3],#[7,3,3],
'filter_w':[7,3,3],#[7,3,3],
'padding':[1,1,1],#[44,42,42],
'stride':[2,2,2],#applied via atrous convs
'pool_size':pool_size,
'output_shape':np.max(cats),#len(np.unique(cats)), #regression or classification (multiple vals)
'la':1, #l2 regularization for FC layer
'dropout_prob':.2,
'channels':num_channels,
'ckpt_dir':'./ckpt_dir',
'model_name':'no_sync_complex',#'complex',#'no_sync_complex', #'complex',#'#real or complex#
'extra_tag':'_' + which_data + '_' + time.strftime('%H_%M_%S'),
'gpu_number':0, #keep at 0 unless using multi-gpu
'restore_model':False,#True,
'height':im_size[0],
'width':im_size[1],
'num_fc':1,
'init':tf.contrib.layers.xavier_initializer_conv2d, #Weight inits #tf.contrib.layers.xavier_initializer
'inner_init':orthogonal_initializer,
'activation':tf.tanh,#complex_tanh,#tf.tanh,# 
'inner_activation':complex_sigmoid,#tf.nn.relu6, #CHANGE HERE
'which_data':which_data
}
s = Map(settings)

# Build Model
session, init_vars, merged, saver, optim, writer, cost, keep_prob, X, targets, Wc, Uc, Wg, Ug, c, state, pred, accuracy = model.build_model(s)
session.run(init_vars)
if s.restore_model == True:
  ckpt = s.ckpt_dir + '/' + s.model_name + s.extra_tag #This won't work... need to add a regexp to grab the most recent model
  ckpts = sorted(glob(ckpt + '/*')) 
  saver.restore(session,ckpts[-1])

#Train model
session, result = model.batch_train(session, merged, saver, optim, writer, cost, keep_prob, accuracy, X, targets, X_train_raw, y_train_temp, X_test_raw, y_test_temp, s)

#Save settings for later
np.savez(meta_output + '/' + s.model_name + s.extra_tag,settings=settings)
