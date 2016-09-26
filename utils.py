import numpy as np
import tensorflow as tf

def orthogonal_initializer(scale = 1.1):
    ''' From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''
    def _initializer(shape, dtype=tf.float32):
      flat_shape = (shape[0], np.prod(shape[1:]))
      a = np.random.normal(0.0, 1.0, flat_shape)
      u, _, v = np.linalg.svd(a, full_matrices=False)
      # pick the one with the correct shape
      q = u if u.shape == flat_shape else v
      q = q.reshape(shape) #this needs to be corrected to float32
      return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
    return _initializer

def slice_for_axis(axis, s):
  return (slice(None),) * axis + (s,)

def fix_complex_gates(a,axis=-1):
  a_shape = a.get_shape()
  la = len(a_shape)
  a_real = a[slice_for_axis(axis=la, s=slice(0, a_shape[la] / 2))]
  a_imag = a[slice_for_axis(axis=la, s=slice(a_shape[la] / 2, None))]
  a_comb = tf.minimum(a_real,a_imag)
  return tf.concat(la,[a_comb,a_comb])

def real_mult(a,b,axis=-1):
  axis = 1
  a_real = a[slice_for_axis(axis=axis, s=slice(0, a.shape[axis] / 2))]
  b_real = b[slice_for_axis(axis=axis, s=slice(0, b.shape[axis] / 2))]
  return (a_real * b_real)

def slice_for_axis(axis, s):
  return (slice(None),) * axis + (s,)

def complex_elemwise_mult(a, b, axis=-1):
  assert a.ndim == b.ndim
  if axis < 0: axis %= a.ndim
  assert 0 <= axis < a.ndim
  #axis = 1 #treat half of the conv filters as complex (for th).  uncomment the above if using tf
  a_shape = a.get_shape()
  la = len(a_shape)
  b_shape = b.get_shape()
  lb = len(b_shape)
  a_real = a[slice_for_axis(axis=la, s=slice(0, a_shape[la] / 2))]
  a_imag = a[slice_for_axis(axis=la, s=slice(a_shape[la] / 2, None))]
  b_real = b[slice_for_axis(axis=lb, s=slice(0, b_shape[lb] / 2))]
  b_imag = b[slice_for_axis(axis=lb, s=slice(b_shape[lb] / 2, None))]
  r_real = tf.mul(a_real,b_real) - tf.mul(a_imag,b_imag)
  r_imag = tf.mul(a_real,b_imag) + tf.mul(a_imag,b_real) #mul versus *?
  return tf.concat(axis,[r_real, r_imag])

def complex_bound(a, axis=-1):
  # Via Associative LSTMs, http://arxiv.org/abs/1602.03032
  if axis < 0: axis %= a.ndim
  assert 0 <= axis < a.ndim
  #axis = 1 #treat half of the conv filters as complex (for th).  uncomment the above if using tf 
  a_shape = a.get_shape()
  la = len(a_shape)
  a_real = a[slice_for_axis(axis=la, s=slice(0, a_shape[la] / 2))]
  a_imag = a[slice_for_axis(axis=la, s=slice(a_shape[la] / 2, None))]
  d = tf.maximum(np.float32(1), tf.sqrt(a_real * a_real + a_imag * a_imag))
  r_real = a_real / d
  r_imag = a_imag / d
  return tf.concat(la,[r_real, r_imag])

def complex_dot(a, b):
  assert a.ndim >= 1
  assert b.ndim >= 1
  a_axis = a.ndim - 1
  a_shape = a.get_shape()
  la = len(a_shape)
  b_shape = b.get_shape()
  lb = len(b_shape)
  a_real = a[slice_for_axis(axis=la, s=slice(0, a_shape[la] / 2))]
  a_imag = a[slice_for_axis(axis=la, s=slice(a_shape[la] / 2, None))]
  b_real = b[slice_for_axis(axis=lb, s=slice(0, b_shape[lb] / 2))]
  b_imag = b[slice_for_axis(axis=lb, s=slice(b_shape[lb] / 2, None))]
  r_real = tf.dot(a_real, b_real) - tf.dot(a_imag, b_imag)
  r_imag = tf.dot(a_real, b_imag) + tf.dot(a_imag, b_real)
  return tf.concat(la,[r_real, r_imag])

