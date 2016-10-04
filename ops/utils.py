import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm
import math

def factors(n):    
    return list(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

def convert_to_phase(ims):
    split_size = ims.shape[-1]//2
    real_ims = ims[:,:,:,:split_size]
    imag_ims = ims[:,:,:,split_size//2:]
    phase_ims = np.zeros((ims.shape[0],ims.shape[1],ims.shape[2],split_size))
    for i in range(split_size):
        for x in range(real_ims.shape[0]):
            for y in range(real_ims.shape[1]):
                phase_ims[x,y,:,i] = math.atan2(np.squeeze(imag_ims[x,y,:,i]),np.squeeze(real_ims[x,y,:,i]))
    return phase_ims

def im_mosaic(ims,phase):
    if ims.shape[-2] == 1:
        pass
    elif ims.shape[-2] > 1:
        ims = ims.reshape(ims.shape[0],ims.shape[1],1,ims.shape[2] * ims.shape[3])
    if phase:
        ims = convert_to_phase(ims)
        cm = 'jet'
    else:
        cm = 'Greys_r'
    sp_factors = factors(ims.shape[-1])
    s1 = sp_factors[np.argmin(np.abs(map(lambda x: x - np.sqrt(ims.shape[-1]),sp_factors)))]
    s2 = ims.shape[-1] // s1
    f = plt.figure()
    for p in tqdm(range(ims.shape[-1])):
        a = plt.subplot(s1,s2,p+1)
	if phase:
            plt.imshow(np.squeeze(ims[:,:,:,p]),cmap=cm, vmin=-3.15,vmax=3.15);
        else:
            plt.imshow(np.squeeze(ims[:,:,:,p]),cmap=cm)
        a.axes.get_xaxis().set_visible(False)
        a.axes.get_yaxis().set_visible(False)
    plt.subplots_adjust(wspace=0.01,hspace=0.01,right=0.8)    
    cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    plt.colorbar(cax=cbar_ax)
    plt.show()

class Map(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.iteritems():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.iteritems():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]

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
  num_filts = a_shape[-1]
  filt_dim = len(a_shape) - 1
  both_splits = tf.split(filt_dim,2,a)
  a_comb = tf.minimum(both_splits[0],both_splits[1]) #real/imaginary
  return tf.concat(filt_dim,[a_comb,a_comb])

def pass_gate(a):
  return a

def real_mult(a,b,axis=-1):
  a_shape = a.get_shape()
  num_filts = a_shape[-1]
  filt_dim = len(a_shape) - 1
  a_real = tf.split(filt_dim,2,a)[0]
  b_real = tf.split(filt_dim,2,b)[0]
  return tf.mul(a_real,b_real)

def slice_for_axis(axis, s):
  return (slice(None),) * axis + (s,)

def complex_elemwise_mult(a, b, axis=-1):
  #axis = 1 #treat half of the conv filters as complex (for th).  uncomment the above if using tf
  a_shape = a.get_shape()
  num_filts = a_shape[-1]
  filt_dim = len(a_shape) - 1
  a_both = tf.split(filt_dim,2,a)
  b_both = tf.split(filt_dim,2,b)
  r_real = tf.mul(a_both[0],b_both[0]) - tf.mul(a_both[1],b_both[1])
  r_imag = tf.mul(a_both[0],b_both[1]) + tf.mul(a_both[1],b_both[0]) #mul versus *?
  return tf.concat(filt_dim,[r_real, r_imag])

def complex_bound(a, axis=-1):
  # Via Associative LSTMs, http://arxiv.org/abs/1602.03032
  #axis = 1 #treat half of the conv filters as complex (for th).  uncomment the above if using tf 
  a_shape = a.get_shape()
  num_filts = a_shape[-1]
  filt_dim = len(a_shape) - 1
  a_both = tf.split(filt_dim,2,a)
  d = tf.maximum(np.float32(1), tf.sqrt(a_both[0] * a_both[0] + a_both[1] * a_both[1]))
  r_real = a_both[0] / d
  r_imag = a_both[1] / d
  return tf.concat(filt_dim,[r_real, r_imag])

def complex_dot_product(a, b):
  a_shape = a.get_shape()
  filt_dim = len(a_shape) - 1
  a_both = tf.split(filt_dim,2,a)
  b_both = tf.split(filt_dim-1,2,b)
  r_real = tf.matmul(a_both[0], b_both[0]) - tf.matmul(a_both[1], b_both[1])
  r_imag = tf.matmul(a_both[0], b_both[1]) + tf.matmul(a_both[1], b_both[0])
  import ipdb;ipdb.set_trace()
  return tf.concat(filt_dim,[r_real, r_imag])

def deconv():
   pass 
