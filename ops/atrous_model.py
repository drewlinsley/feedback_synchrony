from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.models.rnn.ptb import reader
from ops.utils import * #_variable_with_weight_decay
from ops import prepare_mnist_data
from ops import data_loader
import os
import sys
from scipy import sparse 
#Load data
def build_model(s):
	#Initialize some variables
	#if not os.path.exists(s.ckpt_dir):
	#	os.makedirs(s.ckpt_dir)
	#if not os.path.exists('summaries/' + s.model_name + s.extra_tag):
	#	os.makedirs('summaries/' + s.model_name + s.extra_tag)
	prev_channels = s.channels
	prev_height = s.height
	Wi =[]; Ui = []; Wib = []; Uib = []; 
	Wf =[]; Uf = []; Wfb = []; Ufb = []; 
	Wo =[]; Uo = []; Wob = []; Uob = []; 
	Wc =[]; Uc = []; Wcb = []; Ucb = []; 
	Wg =[]; Ug = []; Wgb =[]; Ugb = []; 
	state = [];

	hh = [s.height]
	hc = [prev_channels]
	target_size = [s.batch_size]
	if s.output_shape == 1:
		dt = tf.float32
	else:
		target_size = [s.batch_size,s.output_shape]
		dt = tf.float32#.int64

	if s.model_name == 'complex':
		gate_fun = fix_complex_gates #don't need this if a complex activation function is used...
		mult_fun = complex_elemwise_mult
		dtp_fun = tf.matmul#complex_dot_product
		complex_fun = complex_conv #add synchrony term 
                Sw = [];Sc = [];Sh = [];Scb = [];Shb = [];
	elif s.model_name == 'no_sync_complex':
		gate_fun = fix_complex_gates
		mult_fun = complex_elemwise_mult
		dtp_fun = tf.matmul#complex_dot_product
		#complex_fun = apply_atrous#complex_weight #add synchrony term 
	elif s.model_name == 'real':
		gate_fun = pass_gate
		mult_fun = tf.mul
		dtp_fun = tf.matmul
		#complex_fun = apply_atrous
	else: 
		print('model name is not recognized')
		sys.exit()
 
	with tf.device('/gpu:' + str(s.gpu_number)):
		lr = tf.placeholder(tf.float32, [])
		if s.dropout_prob > 0:
			keep_prob = tf.placeholder(tf.float32)
		else:
			keep_prob = [];
		X = tf.placeholder(tf.float32, [s.batch_size, s.num_steps, s.height, s.width, s.channels]) #batch,time,height,width,channels
		targets = tf.placeholder(dt, target_size) #replace num_steps with 1 if doing a single prediction

		#FCs
		fc_dim = (s.height // s.pool_size) **2 * s.filters[-1]
		fc_weights = []
		fc_biases = []
		for f in range(s.num_fc):
			if f == (s.num_fc-1):
				out_dim = s.output_shape
			else:
				out_dim = fc_dim // 5
			fc_weights.append(tf.Variable(  # fully connected, depth 512.
			tf.truncated_normal([fc_dim, out_dim],stddev= np.sqrt(2 / (s.height * s.width * s.channels)))))
			fc_biases.append(tf.Variable(tf.constant(0.0, shape=[out_dim])))
			fc_dim = out_dim

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
			if s.model_name == 'complex':
				Sw.append(tf.Variable(tf.constant(0.5, shape=[2],dtype=tf.float32),trainable=True,name="Sw_%d" % i))
				Sc.append(tf.get_variable(name="Sc_%d" % i, shape=in_size_u,initializer=s.init()))	
                                Sh.append(tf.get_variable(name="Sh_%d" % i, shape=in_size_w,initializer=s.inner_init()))
                        	Scb.append(tf.Variable(tf.constant(0.0, shape=[s.filters[i]],dtype=tf.float32),trainable=True,name="Scb_%d" % i))
                        	Shb.append(tf.Variable(tf.constant(0.0, shape=[s.filters[i]],dtype=tf.float32),trainable=True,name="Shb_%d" % i))
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
		for time_step in range(s.num_steps):
			h_prev = X[:, time_step, :, :, :]
			for layer in range(num_hidden_layers):
				layer_strides = [1,1,1,1]#HARDCODED FOR HIDDEN np.repeat(s.stride[layer],4).tolist() #4d tensor
				#Facing the input
				xi = tf.nn.atrous_conv2d(h_prev, Wi[layer], s.stride[layer], padding='SAME') + Wib[layer]
				xf = tf.nn.atrous_conv2d(h_prev, Wf[layer], s.stride[layer], padding='SAME') + Wfb[layer]
				xo = tf.nn.atrous_conv2d(h_prev, Wo[layer], s.stride[layer], padding='SAME') + Wob[layer]
				xc = tf.nn.atrous_conv2d(h_prev, Wc[layer], s.stride[layer], padding='SAME') + Wcb[layer]
				#Facing the hidden layer
				hi = tf.nn.atrous_conv2d(state[layer], Ui[layer], s.stride[layer], padding='SAME') + Uib[layer]
				hf = tf.nn.atrous_conv2d(state[layer], Uf[layer], s.stride[layer], padding='SAME') + Ufb[layer]
				ho = tf.nn.atrous_conv2d(state[layer], Uo[layer], s.stride[layer], padding='SAME') + Uob[layer]
				hc = tf.nn.atrous_conv2d(state[layer], Uc[layer], s.stride[layer], padding='SAME') + Ucb[layer]
                                #Synchronize the input and hidden-facing content
                                if s.model_name == 'complex':
                                	xc = complex_fun(xc, Sc[layer], Scb[layer], Sw[layer], s.stride[layer], padding='SAME')
                                	hc = complex_fun(hc, Sh[layer], Shb[layer], Sw[layer], s.stride[layer], padding='SAME')
				#Fix complex gates after applying activations
				i = gate_fun(s.inner_activation(xi + hi)) #need to implement the complex-valued ops fix_complex_gates
				f = gate_fun(s.inner_activation(xf + hf))
				o = gate_fun(s.inner_activation(xo + ho)) #The original implementation handled the h's seperately
				# Feedback gates (apply reichert & serre to the hidden state of the unit:
				target_layer = np.min([layer + s.num_afferents, num_hidden_layers])
				if target_layer > num_hidden_layers - 1:
                                        #gated_prev_timestep = complex_fun(state[layer], Uc[idx], Ucb[idx], Sw[layer], layer_strides, padding='SAME')
					gated_prev_timestep = tf.nn.atrous_conv2d(state[layer], Uc[idx], s.stride[layer], padding='SAME') + Ucb[idx]
					new_c = s.activation((tf.nn.atrous_conv2d(h_prev, Wc[layer], s.stride[layer], padding='SAME') + Wcb[layer]) + gated_prev_timestep)
				else:
					con_range = range(layer,target_layer+1) #need to adjust the hidden state concat
					gates = [s.inner_activation((tf.nn.atrous_conv2d(h_prev, Wg[idx], s.stride[layer], padding='SAME') +  Wgb[idx]) + 
					tf.reduce_sum(tf.reshape((tf.nn.atrous_conv2d(prev_concat_h, Ug[idx], s.stride[layer], padding='SAME') +  Ugb[idx]),(s.batch_size,prev_height,prev_height,s.filters[idx],num_hidden_layers)),4)) for idx in con_range] #restricted to num_afferents levels above the current... is this conv or element
					#Gated_prev_timestep is the activations from all afferents weighted by Gates
					gated_prev_timestep = [gates[idx - layer] * (tf.nn.atrous_conv2d(state[layer], Uc[idx], s.stride[layer], padding='SAME') +  Ucb[idx]) for idx in con_range]
					#c is now calculated as tanh(current hidden content + sum of the gated afferents)
					new_c = s.activation((tf.nn.atrous_conv2d(h_prev, Wc[layer], s.stride[layer], padding='SAME') + Wcb[layer]) + tf.add_n(gated_prev_timestep))
				#Get new h and c as per usual
				c = mult_fun(f, c) + mult_fun(i, new_c) #complex multiplication here
				#state[layer] = mult_fun(o, s.activation(c))
				state[layer] = mult_fun(o, s.activation(c))

			#if connecting each time step to the cost:
			#res_pool_state = tf.reshape(pool_state,[batch_size,prev_height//pool_size*prev_height//pool_size*filters[-1]])
			#logits = tf.nn.bias_add(tf.matmul(state[num_hidden_layers-1], output_weights), output_bias)
			#step_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets[:, time_step])
			#regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases))
			#loss += 5e-4 + regularizers #tf.reduce_sum(step_loss)
			prev_concat_h = tf.concat(3, state)
		pool_state = tf.nn.max_pool(state[layer],ksize=[1,s.pool_size,s.pool_size,1],strides=[1,s.pool_size,s.pool_size,1],padding='VALID',name='end_pool')
		#pool_state = tf.nn.dropout(pool_state,keep_prob) #to add dropout... acting weird tho
		res_pool_state = tf.reshape(pool_state,[s.batch_size,prev_height//s.pool_size*prev_height//s.pool_size*s.filters[-1]])
		#add as many FC layers as you want (need to refactor all this code...)
		for f in range(s.num_fc):
			if f == (s.num_fc - 1): #Prep for classification or regression
				if s.output_shape == 1:
					pred = tf.add(dtp_fun(res_pool_state,fc_weights[f]),fc_biases[f])
					error_loss = tf.reduce_sum((tf.pow(pred-targets, 2))/ s.batch_size)
					error_mean = error_loss
				else:
					pred = tf.add(dtp_fun(res_pool_state,fc_weights[f]),fc_biases[f])
					error_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,targets))
					#pred = tf.nn.softmax(tf.add(dtp_fun(res_pool_state,fc_weights[f]),fc_biases[f]))
					#error_loss = -tf.reduce_sum(targets*tf.log(tf.clip_by_value(pred,1e-10,1.0)))#tf.log(tf.clip_by_value(pred,1e-10,1.0))#tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(pred, targets))
			else:
				res_pool_state = tf.add(dtp_fun(res_pool_state,fc_weights[f]),fc_biases[f]) #Typical FC layer
			regularizers = tf.add(tf.nn.l2_loss(fc_weights[f]),tf.nn.l2_loss(fc_biases[f])) #regularize every layer
		reg_loss = s.la * regularizers
		cost = (error_loss + reg_loss) / tf.cast(s.batch_size,tf.float32)
	tf.scalar_summary("cost", cost)
	if s.output_shape > 1:
		#correct = tf.nn.in_top_k(tf.nn.softmax(pred), targets, 1)
		correct = tf.to_float(tf.equal(tf.argmax(pred,1), tf.argmax(targets,1)))
		#correct = tf.to_float(correct)
		accuracy = tf.reduce_mean(correct)
		tf.scalar_summary("training_accuracy", accuracy)
		tf.scalar_summary("validation_accuracy", accuracy)

	#Prepare session
	tf.image_summary('Wc1',tf.reshape(Wc[0],(s.filters[0],s.filter_r[0],s.filter_r[0],s.channels)))
	train_merged = tf.merge_all_summaries()
	validation_merged = tf.merge_all_summaries()
	merged = [train_merged,validation_merged]
	#config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
	session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

	#Prepare writers
	train_writer = tf.train.SummaryWriter(os.path.join('experiments',s.model_name + s.extra_tag + '_train'), session.graph)
	validation_writer = tf.train.SummaryWriter(os.path.join('experiments',s.model_name + s.extra_tag + '_validation'), session.graph)
	writers = [train_writer,validation_writer]

	# Optimize and initialize model Model
	#optim = tf.train.GradientDescentOptimizer(lr).minimize(cost)
	optim = tf.train.AdamOptimizer(1e-4).minimize(cost)
	#optim = tf.train.AdamOptimizer(1e-4)
	#global_step = tf.Variable(0,name='global_step',trainable=False)
	#grads_and_ars = optimizer.compute_gradients(cost)
	#optim = optim.apply_gradients(grads_and_vars, global_step=global_step)
	saver = tf.train.Saver()
	init_vars = tf.initialize_all_variables()
	#tf.check_numerics()

	return session, init_vars, merged, saver, optim, writers, cost, keep_prob, X, targets, Wc, Wg, Uc, Ug, state, c, pred, accuracy#, keep_prob

def batch_train(session, merged, saver, optim, writer, cost, keep_prob, accuracy, X, targets, X_train_raw, y_train_temp, X_test_raw, y_test_temp, s):
	#Consider clipping
	#grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
	#                                     config.max_grad_norm)
	new_ckpt_dir = s.ckpt_dir + '/' + s.model_name + s.extra_tag + '/'
	if not os.path.exists(new_ckpt_dir):
		os.makedirs(new_ckpt_dir)

	if s.which_data == 'mnist_addition':
		task = prepare_mnist_data.repeat_adding_task
	elif s.which_data == 'coco':
		task = prepare_mnist_data.repeat_task 
	elif s.which_data == 'multi_mnist':
		task = prepare_mnist_data.multi_mnist_adding_task
	elif s.which_data == 'cluttered_mnist_classification':
		task = prepare_mnist_data.cluttered_mnist_classification
        else:
		print('data is not recognized')
		sys.exit()	

	costs = 0.0
	prev_train_costs = 0.0
	prev_val_costs = 0.0
	iters = 0
	val_iters = 0
	data_size = X_train_raw.shape[0]
	val_data_size = X_test_raw.shape[0]
	cv_folds = data_size // s.batch_size
	val_cv_folds = val_data_size // s.batch_size
	train_merged = merged[0]
	valid_merged = merged[1]
	train_writer = writer[0]
	valid_writer = writer[1]
	for i in range(s.epochs):
		print('Epoch', i)
		x,y = task(X_train_raw,y_train_temp,s.num_steps,data_size,[s.height,s.width,s.channels],s.output_shape) #turn regress into a variable passed from main
		vx,vy = task(X_test_raw, y_test_temp,s.num_steps,val_data_size,[s.height,s.width,s.channels],s.output_shape)
		cv_ind = range(data_size)
		np.random.shuffle(cv_ind)
		cv_ind = cv_ind[:(cv_folds*s.batch_size)]
		cv_ind = np.reshape(cv_ind,[cv_folds,s.batch_size])
		val_cv_ind = range(val_data_size)
		np.random.shuffle(val_cv_ind)
		val_cv_ind = val_cv_ind[:(val_cv_folds*s.batch_size)]
		val_cv_ind = np.reshape(val_cv_ind,[val_cv_folds,s.batch_size])
		for idx in range(cv_folds):
			train_idx = cv_ind[idx,:]
			bx = x[train_idx,:,:,:,:]
			by = y[train_idx]
			if sparse.issparse(by):
				by = by.todense()
			if s.dropout_prob > 0:
				result, step_cost, _, train_acc = session.run([train_merged, cost, optim, accuracy],{X: bx, targets: by, keep_prob: s.dropout_prob})
			else:
				result, step_cost, _, train_acc= session.run([train_merged, cost, optim, accuracy],{X: bx, targets: by})
			costs += step_cost
			iters += s.num_steps
			if iters % 10000 == 0:
				#Prepare training stats
				#print(iters, np.exp(costs / iters), train_acc)
				train_writer.add_summary(result, iters)
				train_writer.flush()
				#Prepare validation stats
				val_idx = val_cv_ind[idx%val_cv_folds,:]
				val_labels = vy[val_idx]
				if sparse.issparse(val_labels):
					val_labels = val_labels.todense()
				if s.dropout_prob > 0:
					val_result, val_cost, _, val_acc = session.run([valid_merged, cost, optim, accuracy],
						{X: vx[val_idx,:,:,:,:], targets: val_labels, keep_prob: 1})
				else:
					val_result, val_cost, _, val_acc = session.run([valid_merged, cost, optim, accuracy],
						{X: vx[val_idx,:,:,:,:], targets: val_labels})
				valid_writer.add_summary(val_result,iters)
                                valid_writer.flush()
				sys.stdout.write("training step %d, cost %g, cost delta %g, accuracy %g --- validation cost %g, cost delta %g, accuracy %g     \r"%(iters, np.exp(costs / iters), np.exp((prev_train_costs - costs) / iters), train_acc ,np.exp(val_cost / iters), np.exp((prev_val_costs - costs) / iters), val_acc))
				sys.stdout.flush()
				prev_train_costs = costs
				prev_val_costs = val_cost
				#check_op = tf.add_check_numerics_ops()
				#sess.run([train_merged, check_op])
		checkpoint_file = new_ckpt_dir +  s.model_name + s.extra_tag + '_epoch_' + str(i)
		saver.save(session, checkpoint_file)
