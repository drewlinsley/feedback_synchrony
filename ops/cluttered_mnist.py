#The Cluttered_Mnist generation which is based on the Mnist dataset.
#This Dataset contains the random nums at diff. random locations.

import cPickle
import gzip

import numpy as np
import scipy
from scipy.misc import imsave
from skimage import transform as trf
from skimage.measure import label
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import os
from skimage import transform as tf
from skimage.measure import label, regionprops

def is_empty(any_structure):
    if any_structure:
        return any_structure
    else:
        return (0,0)

def trim_data(random_data):
        label_img = label(random_data>0)
        regions = regionprops(label_img)
        bbox = regions[0].bbox
        return random_data[bbox[0]:bbox[2],bbox[1]:bbox[3]]

def mnist_noise_generation(data):
    noise = []
    data_shape = data.shape
    high_end = 28
    low_end = 24
    data = scipy.misc.imrotate(data,math.degrees(np.random.vonmises(0,0)))
    afine_tf = tf.AffineTransform(shear=np.random.rand())
    data = tf.warp(data,afine_tf)
    start = np.random.randint(8,12)
    stop = np.random.randint(16,20)
    trim_range = range(start,stop)
    data[:start,:] = 0
    data[stop:,:] = 0
    if np.random.rand() < .5:
        data = tf.swirl(data)
    resizer = np.random.randint(low_end,high_end)
    data = scipy.misc.imresize(data,[resizer,resizer])

    sizes = []
    coors = []
    rp = regionprops(label(data>0,connectivity=1))
    for r in rp:
        sizes.append(r.filled_area)
        coors.append(r.coords)
    new_data = np.zeros(data_shape)
    if not sizes:
        pass
    else:
        wo = np.argmax(sizes)
        for i in coors[wo]:
            new_data[i[0],i[1]] = data[i[0],i[1]]
    return new_data.ravel()

#define a function for New Mnist dataset generation

def Data_generation_mnist(data,data_shape,noise_set,board_h,board_w):
   
    #the data size is a Mnist data: 28*28
    if data.shape != (data_shape, data_shape):
        data = data.reshape(data_shape, data_shape)
    
    #define a blackboard
    #define the blackboard whose size is 100*100
    board = np.zeros((board_h,board_w))

    #add noise: here contains 3 diff. noise
    for ns in noise_set:
        if np.sum(np.abs(ns)) > 0:
            random_data = trim_data(ns.reshape(data_shape,data_shape))
            new_shape = random_data.shape
            loc_x = np.random.randint(0,board_h - new_shape[0])
            loc_y = np.random.randint(0,board_w - new_shape[1])
            it_board = board[loc_x:loc_x+new_shape[0],loc_y:loc_y+new_shape[1]]
            board[loc_x:loc_x+new_shape[0],loc_y:loc_y+new_shape[1]] = np.maximum(it_board,random_data)

    #define the diff. locations, here the (x,y) coordinations 
    random_data = trim_data(data)
    new_shape = random_data.shape
    loc_x = np.random.randint(0,board_h - new_shape[0])
    loc_y = np.random.randint(0,board_w - new_shape[1])
    it_board = board[loc_x:loc_x+new_shape[0],loc_y:loc_y+new_shape[1]]
    #inserted the Mnist data, which is the same num. but located in the diff. places
    board[loc_x:loc_x+new_shape[0],loc_y:loc_y+new_shape[1]] = np.maximum(it_board,random_data)
    return board

def get_mnists(N):
    # Load the dataset
    f = gzip.open('data/mnist/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    train_x, train_y = train_set

    h = int(np.sqrt(train_x.shape[1]))
    w = int(np.sqrt(train_x.shape[1]))

    #Generate MNIST noise first]
    new_h = 40
    new_w = 40
    mnist_noise = []
    for k in tqdm(range(0,N)):
        mnist_noise.append(mnist_noise_generation(train_x[np.random.randint(train_x.shape[0])].reshape(h,w)))

    New_Mnist_x = np.zeros((N, new_h, new_w))
    New_Mnist_y = np.zeros((N,))
    how_much_noise = [3,6]
    image_name_index = range(0,N)
    num_ims = train_x.shape[0]
    for k in tqdm(image_name_index):
        # random a data which form the Mnist data train dataset
        i = np.random.randint(0, num_ims)
        random_data = train_x[i]
        random_index = train_y[i]

        #Grab a random number of random noise
        num_noise = np.random.randint(how_much_noise[0], how_much_noise[1])
        noise_set = []
        for i in range(0,num_noise):
            noise_set.append(mnist_noise[np.random.randint(len(mnist_noise))]/255)
        New_Mnist_x[k,:,:] = Data_generation_mnist(random_data, h, noise_set,new_h,new_w) 
        New_Mnist_y[k] = random_index
    return New_Mnist_x, np.asarray(New_Mnist_y)

