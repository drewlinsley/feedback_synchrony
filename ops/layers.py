from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.models.rnn.ptb import reader
from ops.utils import * #_variable_with_weight_decay


def complex_lstm(s):
    

Wi = [tf.Variable(
  tf.random_uniform([emb_size, hidden_size], minval=-init_scale, maxval=init_scale),
  name="Wi_%d" % i) for i in range(num_hidden_layers)]
Ui = [tf.Variable(
  tf.random_uniform([hidden_size, hidden_size], minval=-init_scale, maxval=init_scale),
  name="Ui_%d" % i) for i in range(num_hidden_layers)]
Wf = [tf.Variable(
  tf.random_uniform([emb_size, hidden_size], minval=-init_scale, maxval=init_scale),
  name="Wf_%d" % i) for i in range(num_hidden_layers)]
Uf = [tf.Variable(
  tf.random_uniform([hidden_size, hidden_size], minval=-init_scale, maxval=init_scale),
  name="Uf_%d" % i) for i in range(num_hidden_layers)]
Wc = [tf.Variable(
  tf.random_uniform([emb_size, hidden_size], minval=-init_scale, maxval=init_scale),
  name="Wc_%d" % i) for i in range(num_hidden_layers)]
Uc = [tf.Variable(
  tf.random_uniform([hidden_size, hidden_size], minval=-init_scale, maxval=init_scale),
  name="Uc_%d" % i) for i in range(num_hidden_layers)]
Wo = [tf.Variable(
  tf.random_uniform([emb_size, hidden_size], minval=-init_scale, maxval=init_scale),
  name="Wo_%d" % i) for i in range(num_hidden_layers)]
Uo = [tf.Variable(
  tf.random_uniform([hidden_size, hidden_size], minval=-init_scale, maxval=init_scale),
  name="Uo_%d" % i) for i in range(num_hidden_layers)]
initial_state = tf.zeros([batch_size, hidden_size])
content = initial_state
state = [initial_state] * num_hidden_layers
prev_concat_h = tf.zeros([batch_size, hidden_size * num_hidden_layers])
loss = tf.zeros([])
for time_step in range(num_steps):
  h_prev = X_in[:, time_step, :]
  for layer in range(num_hidden_layers):
    input_gate = tf.nn.sigmoid(tf.matmul(h_prev, Wi[layer])  + tf.matmul(state[layer], Ui[layer]))
    forget_gate = tf.nn.sigmoid(tf.matmul(h_prev, Wf[layer]) + tf.matmul(state[layer], Uf[layer]))
    output_gate = tf.nn.sigmoid(tf.matmul(h_prev, Wo[layer]) + tf.matmul(state[layer], Uo[layer]))
    content = tf.mul(forget_gate,Wc) + tf.mul(input_gate,h_prev)

    state[layer] = tf.mul(output_gate, tf.nn.tanh(content))

    # Main contribution of paper:
    gates = [tf.sigmoid(tf.matmul(h_prev, Wg[i]) + tf.matmul(prev_concat_h, Ug[i])) for i in range(num_hidden_layers)]
    gated_prev_timestep = [gates[i] * tf.matmul(state[layer], Uc[i]) for i in range(num_hidden_layers)]
    new_content = tf.nn.tanh(tf.matmul(h_prev, Wc[layer]) + tf.add_n(gated_prev_timestep))

    content = tf.mul(forget_gate, content) + tf.mul(input_gate, new_content)
    state[layer] = tf.mul(output_gate, tf.nn.tanh(content))

  logits = tf.nn.bias_add(tf.matmul(state[num_hidden_layers-1], output_weights), output_bias)
  step_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets[:, time_step])
  loss += tf.reduce_sum(step_loss)
  prev_concat_h = tf.concat(1, state)

final_state = state

