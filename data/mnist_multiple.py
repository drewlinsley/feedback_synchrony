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

def Data_generation_mnist(data,num_ims,noise_set,data_shape,board_h,board_w):
   
    #the data size is a Mnist data: 28*28
    if data.shape != (data_shape, data_shape):
        data = data.reshape(data_shape, data_shape)
    
    #define a blackboard
    #define the blackboard whose size is 100*100
    board = np.zeros((board_h,board_w))
    diff_size = board_h-data_shape
    #add noise: here contains 3 diff. noise
    for ns in range(0,noise_set):
        i = np.random.randint(0, num_ims)
        random_data = train_x[i].reshape(data_shape,data_shape)
        random_data = trim_data(random_data)
        new_shape = random_data.shape
        loc_x = np.random.randint(0,board_h - new_shape[0])
        loc_y = np.random.randint(0,board_h - new_shape[1])
        it_board = board[loc_x:loc_x+new_shape[0],loc_y:loc_y+new_shape[1]]
        board[loc_x:loc_x+new_shape[0],loc_y:loc_y+new_shape[1]] = np.maximum(it_board,random_data)

    #define the diff. locations, here the (x,y) coordinations 
    l1_x = np.random.randint(0,diff_size)
    l1_y = np.random.randint(0,diff_size)
    
    #inserted the Mnist data, which is the same num. but located in the diff. places
    data = trim_data(data)
    new_shape = data.shape
    board[l1_x:l1_x+new_shape[0],l1_y:l1_y+new_shape[1]] = np.maximum(data,board[l1_x:l1_x+new_shape[0],l1_y:l1_y+new_shape[1]])

    return board

# Load the dataset
f = gzip.open('mnist/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
train_x, train_y = train_set
h = int(np.sqrt(train_x.shape[1]))
w = int(np.sqrt(train_x.shape[1]))

#Generate MNIST noise first]
new_h = 40
new_w = 40
N = 100*1000

#New_data, New_index = Data_generation(random_data, random_index) 
#scipy.misc.imsave('test.jpg', New_data)
output_dir = 'multiple_ims/';
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

New_Mnist_x = np.zeros((N, new_h*new_w))
New_Mnist_y = np.zeros((N,))
how_much_noise = [1,3]
image_name_index = range(0,N)
num_ims = train_x.shape[0]
for k in tqdm(image_name_index):
    # random a data which form the Mnist data train dataset
    i = np.random.randint(0, num_ims)
    random_data = train_x[i]
    random_index = train_y[i]

    #Grab a random number of random noise
    num_noise = np.random.randint(how_much_noise[0], how_much_noise[1])
    New_data = Data_generation_mnist(random_data, num_ims, num_noise, h ,new_h,new_w) 
    New_Mnist_x[k,:] = New_data.reshape(1,new_h*new_w)
    New_Mnist_y[k] = random_index
    imsave(output_dir + str(k) + '_label_' + str(int(random_index)) + '.png',New_data);
    
#print(New_Mnist_x.shape)
#print(New_Mnist_y.shape)

#save the new data and label into "New_Mnist_data.npy" & "New_Mnist_label.npy"
#np.save("Cluttered_Mnist_data.npy", New_Mnist_x)
np.save("Multiple_Mnist_label.npy", New_Mnist_y)

