from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from seaborn import swarmplot
from ops.evaluate_model import evaluate_model
from ops.utils import *
import time

ckpt_dir = 'ckpt_dir/'
meta_dir = 'meta_output/'
model_a = 'no_sync_complex_cluttered_mnist_classification_10_00_25'
model_b = 'real_cluttered_mnist_classification_13_01_26'
a_epoch = 29
b_epoch = 29 #close enough...
comparison_iterations = 1

a_checkpoint = ckpt_dir + model_a + '/' + model_a + '_epoch_' + str(a_epoch)
b_checkpoint = ckpt_dir + model_b + '/' + model_b + '_epoch_' + str(b_epoch)
perf_mat = np.zeros((comparison_iterations,2))
for i in range(comparison_iterations):
	if i == 0:
		perf_mat[i,0], a_preds, x, y, a_sess, a_sav = evaluate_model(meta_dir + model_a + '.npz','','','','',a_checkpoint)
		import ipdb;ipdb.set_trace()
		perf_mat[i,1], b_preds, _, _, b_sess, b_sav = evaluate_model(meta_dir + model_b + '.npz','','',x,y,b_checkpoint)
	else:
		perf_mat[i,0], a_preds, x, y, _, _ = evaluate_model(meta_dir + model_a + '.npz',a_sess,a_sav,a_checkpoint)
		perf_mat[i,1], b_preds, _, _, _, _ = evaluate_model(meta_dir + model_b + '.npz',b_sess,b_sav,x,y,b_checkpoint)

#Swarmplot here
np.save('comparison',perf_mat)