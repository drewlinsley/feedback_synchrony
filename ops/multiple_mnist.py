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

def Data_generation_mnist(data,data_shape,board_h,board_w):
   
    #the data size is a Mnist data: 28*28
    #if data.shape != (data_shape, data_shape):
    #    data = data.reshape(data_shape, data_shape)
    
    #define a blackboard
    #define the blackboard whose size is 100*100
    board = np.zeros((board_h,board_w))
    diff_size = board_h-data_shape
    #add noise: here contains 3 diff. noise
    for ns in range(0,len(data)):
        random_data = data[ns].reshape(data_shape,data_shape)
        random_data = trim_data(random_data)
        new_shape = random_data.shape
        loc_x = np.random.randint(0,board_h - new_shape[0])
        loc_y = np.random.randint(0,board_h - new_shape[1])
        it_board = board[loc_x:loc_x+new_shape[0],loc_y:loc_y+new_shape[1]]
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
    #N = 100*1000

    New_Mnist_x = np.zeros((N, new_h, new_w))
    New_Mnist_y = []
    how_much_noise = [2,5]
    image_name_index = range(0,N)
    num_ims = train_x.shape[0]
    for k in tqdm(image_name_index):
        # random a data which form the Mnist data train dataset
        it_num_noise = np.random.randint(how_much_noise[0],how_much_noise[1],1)
        i = np.random.randint(0, num_ims,it_num_noise)
        random_data = train_x[i]
        random_index = train_y[i]

        #Grab a random number of random noise
        New_Mnist_x[k,:,:] = Data_generation_mnist(random_data, h ,new_h,new_w) 
        New_Mnist_y.append(np.sum(random_index))
    return New_Mnist_x, np.asarray(New_Mnist_y)
