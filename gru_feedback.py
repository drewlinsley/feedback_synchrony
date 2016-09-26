import numpy as np
import tensorflow as tf
from tensorflow.models.rnn.ptb import reader
train_data, valid_data, test_data, vocab = reader.ptb_raw_data('data/')


# Hyperparameters
batch_size = 20
num_steps = 20
hidden_size = 200
emb_size = 200 # Note: this is kind of a cheat. This will *not* work if emb_size != hidden_size
vocab_size = 10000
epochs = 2
init_scale = 0.1
num_hidden_layers = 3

lr = tf.placeholder(tf.float32, [])

## Build Model
session = tf.Session()

X = tf.placeholder(tf.int32, [batch_size, num_steps])
targets = tf.placeholder(tf.int64, [batch_size, num_steps])

embedding = tf.Variable(
  tf.random_uniform([vocab_size, emb_size], minval=-init_scale, maxval=init_scale),
  name="embedding")

# For input gate.
Wi = [tf.Variable(
  tf.random_uniform([emb_size, hidden_size], minval=-init_scale, maxval=init_scale),
  name="Wi_%d" % i) for i in range(num_hidden_layers)]
Ui = [tf.Variable(
  tf.random_uniform([hidden_size, hidden_size], minval=-init_scale, maxval=init_scale),
  name="Ui_%d" % i) for i in range(num_hidden_layers)]

# For forget gate.
Wf = [tf.Variable(
  tf.random_uniform([emb_size, hidden_size], minval=-init_scale, maxval=init_scale),
  name="Wf_%d" % i) for i in range(num_hidden_layers)]
Uf = [tf.Variable(
  tf.random_uniform([hidden_size, hidden_size], minval=-init_scale, maxval=init_scale),
  name="Uf_%d" % i) for i in range(num_hidden_layers)]

# For content -- Quick note: there's no transformation from content -> state. They are both
# the same size.
Wc = [tf.Variable(
  tf.random_uniform([emb_size, hidden_size], minval=-init_scale, maxval=init_scale),
  name="Wc_%d" % i) for i in range(num_hidden_layers)]
Uc = [tf.Variable(
  tf.random_uniform([hidden_size, hidden_size], minval=-init_scale, maxval=init_scale),
  name="Uc_%d" % i) for i in range(num_hidden_layers)]

# For hidden state output gate.
Wo = [tf.Variable(
  tf.random_uniform([emb_size, hidden_size], minval=-init_scale, maxval=init_scale),
  name="Wo_%d" % i) for i in range(num_hidden_layers)]
Uo = [tf.Variable(
  tf.random_uniform([hidden_size, hidden_size], minval=-init_scale, maxval=init_scale),
  name="Uo_%d" % i) for i in range(num_hidden_layers)]

# For gated feedback gates (e.g. the contribution of the paper).
Wg = [tf.Variable(
  tf.random_uniform([emb_size, 1], minval=-init_scale, maxval=init_scale),
  name="Wg_%d" % i) for i in range(num_hidden_layers)]
Ug = [tf.Variable(
  tf.random_uniform([hidden_size * num_hidden_layers, 1], minval=-init_scale, maxval=init_scale),
  name="Ug_%d" % i) for i in range(num_hidden_layers)]

# For output.
output_weights = tf.Variable(
  tf.random_uniform([hidden_size, vocab_size], minval=-init_scale, maxval=init_scale),
  name="output_weights")
output_bias = tf.Variable(tf.zeros([vocab_size]), name="output_bias")

X_in = tf.nn.embedding_lookup(embedding, X)

initial_state = tf.zeros([batch_size, hidden_size])
content = initial_state
state = [initial_state] * num_hidden_layers
prev_concat_h = tf.zeros([batch_size, hidden_size * num_hidden_layers])
loss = tf.zeros([])
# TODO: prev concat h
for time_step in range(num_steps):
  h_prev = X_in[:, time_step, :]
  for layer in range(num_hidden_layers):
    input_gate = tf.nn.sigmoid(tf.matmul(h_prev, Wi[layer])  + tf.matmul(state[layer], Ui[layer]))
    forget_gate = tf.nn.sigmoid(tf.matmul(h_prev, Wf[layer]) + tf.matmul(state[layer], Uf[layer]))
    output_gate = tf.nn.sigmoid(tf.matmul(h_prev, Wo[layer]) + tf.matmul(state[layer], Uo[layer]))
    
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
cost = loss / batch_size

tf.scalar_summary("cost", cost)
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("summaries/gfrnn", session.graph_def)

# Train Model
session.run(tf.initialize_all_variables())
sgd = tf.train.GradientDescentOptimizer(lr).minimize(cost)
costs = 0.0
iters = 0
for i in range(epochs):
  print 'Epoch', i
  for step, (x, y) in enumerate(reader.ptb_iterator(train_data, batch_size, num_steps)):
    result, step_cost, _, = session.run([merged, cost, sgd],
                             {X: x, targets: y, lr: 1.0 / (i + 1)})
    costs += step_cost
    iters += num_steps
    if iters % 1000 == 0:
      print iters, np.exp(costs / iters)
      writer.add_summary(result, iters)
      writer.flush()
