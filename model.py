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
def build_model(s):
	#Initialize some variables
	if not os.path.exists(s.ckpt_dir):
	    os.makedirs(s.ckpt_dir)
	prev_channels = s.channels
	prev_height = s.height
	Wi =[]; Ui = []; Wib = []; Uib = []; 
	Wf =[]; Uf = []; Wfb = []; Ufb = []; 
	Wo =[]; Uo = []; Wob = []; Uob = []; 
	Wc =[]; Uc = []; Wcb = []; Ucb = []; 
	Wg =[]; Ug = []; Wgb =[]; Ugb = []; 
	Sc = []; Sh = []; state = [];

	hh = [s.height]
	hc = [prev_channels]

	if s.model_name == 'complex':
		gate_fun = fix_complex_gates
		mult_fun = complex_elemwise_mult
	elif s.model_name == 'real':
		gate_fun = pass_gate
		mult_fun = tf.mul
	else: 
		print('model name is not recognized')
		sys.exit()
	with tf.device('/gpu:' + str(s.gpu_number)):
	  lr = tf.placeholder(tf.float32, [])
	  #keep_prob = tf.placeholder(tf.float32)
	  X = tf.placeholder(tf.float32, [s.batch_size, s.num_steps, s.height, s.width, s.channels]) #batch,time,height,width,channels
	  targets = tf.placeholder(tf.float32, [s.batch_size]) #replace num_steps with 1 if doing a single prediction

	  #FC 
	  fc1_weights = tf.Variable(  # fully connected, depth 512.
	      tf.truncated_normal([s.FC_dim, s.output_shape],
	                          stddev=1.0)) #used to be 0.1 for some reason
	  fc1_biases = tf.Variable(tf.constant(0.0, shape=[1]))

	  #conv lstm
	  for i in range(0,len(s.filters)):
	    in_size_w = [s.filter_r[i], s.filter_w[i], prev_channels, s.filters[i]]
	    in_size_u = [s.filter_r[i], s.filter_w[i], s.filters[i],s.filters[i]]

	    # I
	    Wi.append(tf.get_variable(name="Wi_%d" % i, shape=in_size_w,initializer=s.init()))
	    Ui.append(tf.get_variable(name="Ui_%d" % i, shape=in_size_u,initializer=s.inner_init()))
	    Wib.append(tf.Variable(tf.constant(0.0, shape=[s.filters[i]],dtype=tf.float32),trainable=True,name="Wib_%d" % i))
	    Uib.append(tf.Variable(tf.constant(0.0, shape=[s.filters[i]],dtype=tf.float32),trainable=True,name="Uib_%d" % i))
	    # F
	    Wf.append(tf.get_variable(name="Wf_%d" % i, shape=in_size_w,initializer=s.init()))
	    Uf.append(tf.get_variable(name="Uf_%d" % i, shape=in_size_u,initializer=s.inner_init()))
	    Wfb.append(tf.Variable(tf.constant(0.0, shape=[s.filters[i]],dtype=tf.float32),trainable=True,name="Wfb_%d" % i))
	    Ufb.append(tf.Variable(tf.constant(0.0, shape=[s.filters[i]],dtype=tf.float32),trainable=True,name="Ufb_%d" % i))
	    # C
	    Wc.append(tf.get_variable(name="Wc_%d" % i, shape=in_size_w,initializer=s.init()))
	    Uc.append(tf.get_variable(name="Uc_%d" % i, shape=in_size_u,initializer=s.inner_init()))
	    Wcb.append(tf.Variable(tf.constant(0.0, shape=[s.filters[i]],dtype=tf.float32),trainable=True,name="Wcb_%d" % i))
	    Ucb.append(tf.Variable(tf.constant(0.0, shape=[s.filters[i]],dtype=tf.float32),trainable=True,name="Ucb_%d" % i))
	    # O
	    Wo.append(tf.get_variable(name="Wo_%d" % i, shape=in_size_w,initializer=s.init()))
	    Uo.append(tf.get_variable(name="Uo_%d" % i, shape=in_size_u,initializer=s.inner_init()))
	    Wob.append(tf.Variable(tf.constant(0.0, shape=[s.filters[i]],dtype=tf.float32),trainable=True,name="Wob_%d" % i))
	    Uob.append(tf.Variable(tf.constant(0.0, shape=[s.filters[i]],dtype=tf.float32),trainable=True,name="Uob_%d" % i))
	    # Gated feedback (ala Bengio group, 2015)
	    Wg.append(tf.get_variable(name="Wg_%d" % i, shape=in_size_w, initializer=s.init())) #vector with current filter size... scalar weight on activation maps need to reshape into a tensor
	    Ug.append(tf.get_variable(name="Ug_%d" % i, shape=[s.filter_r[i], s.filter_w[i], np.sum(s.filters),np.sum(s.filters)],initializer=s.inner_init())) #vector with num_filters size... scalar weight on activation maps need to reshape into a tensor.... accepts all layers... fix this
	    Wgb.append(tf.Variable(tf.constant(0.0, shape=[s.filters[i]],dtype=tf.float32),trainable=True,name="Wgb_%d" % i))
	    Ugb.append(tf.Variable(tf.constant(0.0, shape=[np.sum(s.filters)],dtype=tf.float32),trainable=True,name="Ugb_%d" % i))
	    prev_channels = s.filters[i]
	    # Add synchrony here
	    #Sc.append(tf.Variable(tf.constant(0.5, shape=[2],dtype=tf.float32),trainable=True,name="Sc_%d" % i))
	    #Sh.append(tf.Variable(tf.constant(0.5, shape=[2],dtype=tf.float32),trainable=True,name="Sh_%d" % i))
	    # Preallocate the state
	    state.append(tf.zeros([s.batch_size,prev_height,prev_height,prev_channels])) #Consider concatenating Weight matrices... Should be much faster
	    prev_height = s.height #this is funky... change to something adaptive once I allow for different sized layers (via deconv)
	    prev_channels = s.channels
	    hh.append(prev_height)
	    hc.append(prev_channels)
	  # For the output
	  output_weights = tf.get_variable(name="output_weights", shape=[s.filters[-1], s.output_shape], initializer=s.inner_init())
	  output_bias = tf.Variable(tf.zeros([s.output_shape]), name="output_bias")

	  #Start the RNN loop
	  num_hidden_layers = len(s.filters)
	  c = state[0]
	  prev_concat_h = tf.zeros([s.batch_size,prev_height,prev_height,np.sum(s.filters)])#for now all filters have to be the same size... in the future use up/downsampling
	  loss = tf.zeros([])
	  # TODO: prev concat h
	  for time_step in range(s.num_steps):
	    h_prev = X[:, time_step, :, :, :]
	    for layer in range(num_hidden_layers):
	      layer_strides = np.repeat(s.stride[layer],4).tolist() #4d tensor
	      #Facing the input
	      xi = tf.nn.conv2d(h_prev, Wi[layer], layer_strides, padding='SAME') + Wib[layer]
	      xf = tf.nn.conv2d(h_prev, Wf[layer], layer_strides, padding='SAME') + Wfb[layer]
	      xo = tf.nn.conv2d(h_prev, Wo[layer], layer_strides, padding='SAME') + Wob[layer]
	      xc = tf.nn.conv2d(h_prev, Wc[layer], layer_strides, padding='SAME') + Wcb[layer]
	      #Facing the hidden layer
	      hi = tf.nn.conv2d(state[layer], Ui[layer], layer_strides, padding='SAME') + Uib[layer]
	      hf = tf.nn.conv2d(state[layer], Uf[layer], layer_strides, padding='SAME') + Ufb[layer]
	      ho = tf.nn.conv2d(state[layer], Uo[layer], layer_strides, padding='SAME') + Uob[layer]
	      hc = tf.nn.conv2d(state[layer], Uc[layer], layer_strides, padding='SAME') + Ucb[layer]
	      #Fix complex gates after applying activations
	      i = gate_fun(s.inner_activation(xi + hi)) #need to implement the complex-valued ops fix_complex_gates
	      f = gate_fun(s.inner_activation(xf + hf))
	      o = gate_fun(s.inner_activation(xo + ho)) #The original implementation handled the h's seperately

	      # Main contribution of paper:
	      target_layer = np.min([layer + s.num_afferents, num_hidden_layers])
	      if layer == target_layer:
	        gated_prev_timestep = tf.nn.conv2d(state[layer], Uc[idx], layer_strides, padding='SAME') + Ucb[idx]
	        new_c = s.activation((tf.nn.conv2d(h_prev, Wc[layer], layer_strides, padding='SAME') + Wcb[layer]) + gated_prev_timestep)
	      else:
	        con_range = range(layer,target_layer) #need to adjust the hidden state concat
	        gates = [s.inner_activation((tf.nn.conv2d(h_prev, Wg[idx], layer_strides, padding='SAME') +  Wgb[idx]) + 
	          tf.reduce_sum(tf.reshape((tf.nn.conv2d(prev_concat_h, Ug[idx], layer_strides, padding='SAME') +  Ugb[idx]),(s.batch_size,prev_height,prev_height,s.filters[idx],num_hidden_layers)),4)) for idx in con_range] #restricted to num_afferents levels above the current... is this conv or element
	        #Gated_prev_timestep is the activations from all afferents weighted by Gates
	        gated_prev_timestep = [gates[idx - layer] * (tf.nn.conv2d(state[layer], Uc[idx], layer_strides, padding='SAME') +  Ucb[idx])for idx in con_range]
	        #c is now calculated as tanh(current hidden content + sum of the gated afferents)
	        new_c = s.activation((tf.nn.conv2d(h_prev, Wc[layer], layer_strides, padding='SAME') + Wcb[layer]) + tf.add_n(gated_prev_timestep))
	      #Get new h and c as per usual
	      c = mult_fun(f, c) + mult_fun(i, new_c) #complex multiplication here
	      state[layer] = mult_fun(o, s.activation(c))

	    #if classsificaiton
	    #res_pool_state = tf.reshape(pool_state,[batch_size,prev_height//pool_size*prev_height//pool_size*filters[-1]])
	    #logits = tf.nn.bias_add(tf.matmul(state[num_hidden_layers-1], output_weights), output_bias)
	    #step_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets[:, time_step])
	    #regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases))
	    #loss += 5e-4 + regularizers #tf.reduce_sum(step_loss)
	    prev_concat_h = tf.concat(3, state)
	  pool_state = tf.nn.max_pool(state[layer],ksize=[1,s.pool_size,s.pool_size,1],strides=[1,s.pool_size,s.pool_size,1],padding='VALID',name='end_pool')
	  #pool_state = tf.nn.dropout(pool_state,keep_prob) #to add dropout... acting weird tho
	  res_pool_state = tf.reshape(pool_state,[s.batch_size,prev_height//s.pool_size*prev_height//s.pool_size*s.filters[-1]])
	  pred = tf.add(tf.matmul(res_pool_state,fc1_weights),fc1_biases)
	  regularizers = tf.add(tf.nn.l2_loss(fc1_weights),tf.nn.l2_loss(fc1_biases))
	  error_loss = tf.reduce_sum((tf.pow(pred-targets, 2))/ s.batch_size)
	  reg_loss = s.la * regularizers
	  cost = error_loss + reg_loss  

	tf.scalar_summary("cost", cost)
	tf.image_summary('Wc1',tf.reshape(Wc[0],(s.filters[0],s.filter_r[0],s.filter_r[0],s.channels)))
	merged = tf.merge_all_summaries()
	#config = tf.ConfigProto()
	#config.gpu_options.allow_growth = True
	#config.log_device_placement=True
	#session = tf.Session(config)
	session = tf.Session()
	writer = tf.train.SummaryWriter('summaries/' + s.model_name, session.graph)

	# Train Model
	#optim = tf.train.GradientDescentOptimizer(lr).minimize(cost)
	optim = tf.train.AdamOptimizer(1e-4).minimize(cost)
	saver = tf.train.Saver()
	init_vars = tf.initialize_all_variables()

	return session, init_vars, merged, saver, optim, writer, cost, X, targets#, keep_prob


def batch_train(session, merged, saver, optim, writer, cost, X, targets, X_train_raw, y_train_temp, s):
	#Consider clipping
	#grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
	#                                     config.max_grad_norm)
	costs = 0.0
	iters = 0
	data_size = X_train_raw.shape[0]
	cv_folds = data_size // s.batch_size
	for i in range(s.epochs):
	  print('Epoch', i)
	  x,y = prepare_mnist_data.adding_task(X_train_raw,y_train_temp,s.num_steps,data_size,[s.height,s.width],'regress') #turn regress into a variable passed from main
	  cv_ind = range(data_size)
	  np.random.shuffle(cv_ind)
	  cv_ind = np.reshape(cv_ind,[cv_folds,s.batch_size])
	  for idx in range(cv_folds):
	    train_idx = cv_ind[idx,:]
	    bx = x[train_idx,:,:,:,:]
	    by = y[train_idx]
	    result, step_cost, _, = session.run([merged, cost, optim],
	                           #{X: x, targets: y, lr: 1.0 / (i + 1)})
	                           #{X: bx, targets: by, keep_prob: s.dropout_prob})
	                           {X: bx, targets: by})
	    costs += step_cost
	    iters += s.num_steps
	    if iters % 10000 == 0:
	      print(iters, np.exp(costs / iters))
	      writer.add_summary(result, iters)
	      writer.flush()
	  checkpoint_file = s.ckpt_dir + '/' + s.model_name + '_epoch_' + str(i)
	  saver.save(session, checkpoint_file)
