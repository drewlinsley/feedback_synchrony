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

def pass_gate(a,b=None):
    return a

def slice_mult(a,b,s=0):
    a_shape = a.get_shape()
    num_filts = a_shape[-1]
    filt_dim = len(a_shape) - 1
    #a_real = tf.split(filt_dim,2,a)[s]
    b_slice = tf.split(filt_dim,2,b)[s]
    return tf.mul(a,b_slice)

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
    return tf.concat(filt_dim,[r_real, r_imag])

def upsample(image,in_size,out_size,align_corners=False):
    if in_size == out_size:
        out = pass_gate
    else:
        out = tf.image.resize_nearest_neighbor(image,size,align_corners)
    return out

def create_deconv(im_size,filters,fsize):
    #####
    # TH input shape: (samples, input_depth, rows, cols)
    # TH kernel shape: (depth, input_depth, rows, cols)
    x = T.tensor4('x')
    kernel = T.tensor4('kernel')
    b = T.vector('b')
    half_filters = filters//2
    conv_out = T.nnet.conv2d(x, kernel,
             border_mode='valid',
             subsample=(1, 1),
             filter_flip=False,  # <<<<< IMPORTANT 111, dont flip kern
             input_shape=(1,half_filters,im_size[0],im_size[1]),
             filter_shape=(1,half_filters,fsize,fsize))
    #output = conv_out + K.reshape(b, (1, 1, 1, 1))
    deconv_fun = theano.function([x,kernel],conv_out)
    return deconv_fun


def deconv(my_kernel,my_input_im,my_bias,deconv_fun):
    num_filters = my_kernel.shape[1]
    half_filters = num_filters//2
    real_kernel = my_kernel[:,:half_filters,:,:]
    im_kernel = my_kernel[:,half_filters:,:,:]
    real_input_im = my_input_im[:,:half_filters,:,:] * my_bias[:half_filters]
    im_input_im = my_input_im[:,half_filters:,:,:] * my_bias[half_filters:]
    real_d_im = np.squeeze(deconv_fun(real_input_im,real_kernel))
    im_d_im = np.squeeze(deconv_fun(im_input_im,im_kernel))

    phase_im = real_d_im
    for x in range(real_d_im.shape[0]):
        for y in range(real_d_im.shape[1]):
            phase_im[x,y] = math.degrees(math.atan2(im_d_im[x,y],phase_im[x,y]))

    return phase_im

def layer_activations(seq,X_test,layer_number_in,layer_number_out,batch_index):
    if layer_number_in != layer_number_out:
        get_layer_output = K.function([seq.layers[layer_number_in].input,K.learning_phase()],[seq.layers[layer_number_out].output])
    else:
        get_layer_output = K.function([seq.layers[layer_number_in].input],[seq.layers[layer_number_out].output])
    small_batch = X_test[batch_index,:,:,:,:]
    small_batch_single = np.repeat(small_batch[:,0,:,:,:],small_batch.shape[1],axis=1)
    small_batch = np.concatenate((small_batch,small_batch_single[:,:,None,:,:]),axis=0)
    mnist_activations = get_layer_output([small_batch])[0]
    return get_layer_output, small_batch, mnist_activations

def process_activations(mnist_activations,seq,layer_number_in,layer_number_out,image_number,timepoint):
    my_input_im = np.squeeze(mnist_activations[image_number,timepoint,:,:,:])[None,:,:,:]
    my_kernel = seq.layers[layer_number_out].U_c.get_value()[0][None,:,:,:]
    my_b =  seq.layers[layer_number_out].get_weights()[8]
    return my_input_im, my_kernel, my_b



def real_mul(a,b):
    b_shape = b.get_shape()
    filt_dim = len(b_shape) - 1
    rc = tf.split(filt_dim,2,b)
    return tf.concat(filt_dim,[tf.mul(a,rc[0]),rc[1]])

def phase(act):
    a_shape = act.get_shape()
    filt_dim = len(a_shape) - 1
    a = tf.split(filt_dim,2,act)
    return atan2(a[1],a[0])

def modulus(act):
    a_shape = act.get_shape()
    filt_dim = len(a_shape) - 1
    a = tf.split(filt_dim,2,act) 
    return tf.sqrt(tf.add(tf.square(a[0]),tf.square(a[1]))) 

def convert_to_complex(modulus,phase):
    return tf.mul(modulus,tf.exp(phase))

def complex_comb(cplx_act,synch):
    p = phase(cplx_act)
    m = modulus(cplx_act)
    return synchronize(m,cplx_act,synch)

def synchronize(modulus,cmplx,sy):
    c_shape = cmplx.get_shape()
    filt_dim = len(c_shape) - 1
    rc = tf.split(filt_dim,2,cmplx) #real part of the complex valued weights
    chi = tf.scalar_mul(sy[0],tf.add(modulus,rc[0])) #"classic" term
    delta = tf.scalar_mul(sy[1],rc[1]) #"synchrony" term
    return tf.concat(filt_dim,[chi,delta])#tf.concat(filt_dim,[chi,rc[1]])
    #return tf.add(tf.mul(re,sy[0]),tf.mul(cmplx,sy[1])) #weighted combo. can learn this in the future

def atan2(y, x):
    angle = tf.select(tf.greater(x,0.0), tf.atan(y/x) + np.pi, tf.zeros_like(x))
    angle = tf.select(tf.logical_and(tf.less(x,0.0),  tf.greater_equal(y,0.0)), tf.atan(y/x) + np.pi, angle)
    angle = tf.select(tf.logical_and(tf.less(x,0.0),  tf.less(y,0.0)), tf.atan(y/x) - np.pi, angle)
    angle = tf.select(tf.logical_and(tf.equal(x,0.0), tf.greater(y,0.0)), 0.5*np.pi * tf.ones_like(x), angle)
    angle = tf.select(tf.logical_and(tf.equal(x,0.0), tf.less(y,0.0)), -0.5*np.pi * tf.ones_like(x), angle)
    angle = tf.select(tf.logical_and(tf.equal(x,0.0), tf.equal(y,0.0)), np.nan * tf.zeros_like(x), angle)
    return angle

def complex_arg(cmplx):
    c_shape = cmplx.get_shape()
    filt_dim = len(c_shape) - 1
    rc = tf.split(filt_dim,2,cmplx) #real part of the complex valued weights
    return atan2(rc[1],rc[0])

def complex_activation(f,z):
    return tf.concat(len(z.get_shape())-1,[slice_mult(tf.truediv(f(modulus(z)) , modulus(z)),z,0),slice_mult(tf.truediv(f(modulus(z)) , modulus(z)),z,1)])

def complex_sigmoid(z):
    return complex_activation(tf.sigmoid,z)

def complex_tanh(z):
    return complex_activation(tf.tanh,z)

def complex_conv(Z,W,b,sw,stride,padding='SAME'):
    m = modulus(Z)
    synchrony = tf.scalar_mul(sw[0],tf.add(tf.nn.conv2d(Z,W,stride,padding=padding),b))
    classic = tf.scalar_mul(sw[1],tf.add(tf.nn.conv2d(tf.concat(len(Z.get_shape())-1,[m,m]),W,stride,padding=padding),b))
    return tf.add(synchrony,classic) 

