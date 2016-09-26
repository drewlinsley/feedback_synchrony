import numpy as np
from glob import glob
from scipy import misc
import re
#from keras.datasets import mnist
from tqdm import tqdm

def train(which_data='cluttered_mnist'):
    if which_data == 'adding_mnist':
        data = np.load('/home/drew/Documents/addition_cluttered_mnist_32_32.npz');
        X_train_raw = data['X_train_raw']
        y_train_temp = data['y_train_temp']
        X_test_raw = data['X_test_raw']
        y_test_temp = data['y_test_temp']
        train_num = data['train_num']
        im_size = data['im_size']
    elif which_data == 'mnist':
        (X_train_raw, y_train_temp), (X_test_raw, y_test_temp) = mnist.load_data()
        train_num = X_train_raw.shape[0]
        im_size = X_train_raw.shape[-2:]
    elif which_data == 'cluttered_mnist':
        from scipy import misc
        from glob import glob
        import re
        file_dir = 'data/cluttered_ims'
        ims = glob(file_dir +'/*.png')
        im_array = []
        im_idx = []
        for i in tqdm(range(len(ims))):
            it_name = ims[i]
            im_array.append(misc.imread(it_name))
            im_idx.append(int(re.split('.png',re.split('_',it_name)[-1])[0]))
        im_array = np.asarray(im_array)
        im_idx = np.asarray(im_idx)
        X_train_raw = im_array[0:int(np.round(0.9*len(im_array))),:,:]
        y_train_temp = im_idx[0:int(np.round(0.9*len(im_array)))]
        X_test_raw = im_array[int(np.round(0.9*len(im_array)))::,:,:]
        y_test_temp = im_idx[int(np.round(0.9*len(im_array)))::]
        train_num = X_train_raw.shape[0]
        im_size = X_train_raw.shape[-2:]
    elif which_data == 'multi_mnist':
        from scipy import misc
        from glob import glob
        import re
        file_dir = '/home/rithesh/Documents/tf/data/multiple_ims'
        ims = glob(file_dir +'/*.png')
        im_array = []
        im_idx = []
        for i in tqdm(range(len(ims))):
            it_name = ims[i]
            im_array.append(misc.imresize(misc.imread(it_name),[40,40]))
            im_idx.append(int(re.split('ims/',re.split('.png',re.split('_',it_name)[-1])[0])[-1][0]))
        im_array = np.asarray(im_array)
        im_idx = np.asarray(im_idx)
        X_train_raw = im_array[0:int(np.round(0.9*len(im_array))),:,:]
        y_train_temp = im_idx[0:int(np.round(0.9*len(im_array)))]
        X_test_raw = im_array[int(np.round(0.9*len(im_array)))::,:,:]
        y_test_temp = im_idx[int(np.round(0.9*len(im_array)))::]
        train_num = X_train_raw.shape[0]
        im_size = X_train_raw.shape[-2:]

    return X_train_raw,y_train_temp,X_test_raw,y_test_temp,train_num,im_size

def test(X_test_raw,y_test_temp,examplesPer,maxToAdd,channels,im_size,classify_or_regress):
    y_test        = []    
    X_test     = np.zeros((examplesPer,maxToAdd,channels,im_size[0],im_size[1]))
    for i in range(0,examplesPer):
        output      = np.zeros((maxToAdd,channels,im_size[0],im_size[1]))
        numToAdd    = np.ceil(np.random.rand()*maxToAdd)
        indices     = np.random.choice(X_test_raw.shape[0],size=numToAdd)
        example     = X_test_raw[indices]
        exampleY    = y_test_temp[indices]
        output[0:numToAdd,0,:,:] = example
        X_test[i,:,:,:,:] = output
        y_test.append(np.sum(exampleY))

    X_test  = np.array(X_test)
    y_test  = np.array(y_test)       
    if classify_or_regress == 'classify':
        y_test = np.equal.outer(y_test, np.arange(maxToAdd * 9)).astype(np.float)
    
    return X_test,y_test
