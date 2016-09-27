import numpy as np



def adding_task(X_train_raw,y_train_temp,maxToAdd,examplesPer,im_size,classify_or_regress):
    X_train       = []
    y_train       = []

    X_train     = np.zeros((examplesPer,maxToAdd,im_size[0],im_size[1],1))

    for i in range(0,examplesPer):
        #initialize a training example of max_num_time_steps,im_size,im_size
        output      = np.zeros((maxToAdd,im_size[0],im_size[1],1))
        #decide how many MNIST images to put in that tensor
        numToAdd    = np.ceil(np.random.rand()*maxToAdd)
        #sample that many images
        indices     = np.random.choice(X_train_raw.shape[0],size=int(numToAdd))
        example     = X_train_raw[indices]
        #sum up the outputs for new output
        exampleY    = y_train_temp[indices]
        output[0:numToAdd,:,:,0] = example
        X_train[i,:,:,:,:] = output
        y_train.append(np.sum(exampleY))

    y_train     = np.array(y_train)
    if classify_or_regress == 'classify':
        y_train = np.equal.outer(y_train, np.arange(maxToAdd * 9)).astype(np.float)
    return X_train, y_train

def repeat_adding_task(X_train_raw,y_train_temp,maxToAdd,examplesPer,im_size,classify_or_regress):
    X_train       = []
    y_train       = []

    X_train     = np.zeros((examplesPer,maxToAdd,im_size[0],im_size[1],1))

    for i in range(0,examplesPer):
        #initialize a training example of max_num_time_steps,im_size,im_size
        output      = np.zeros((maxToAdd,im_size[0],im_size[1],1))
        #decide how many MNIST images to put in that tensor
        numToAdd    = np.ceil(np.random.rand()*maxToAdd)
        #sample that many images
        indices     = np.repeat(np.random.choice(X_train_raw.shape[0],size=1),maxToAdd)
        example     = X_train_raw[indices]
        #sum up the outputs for new output
        exampleY    = y_train_temp[indices]
        output[0:numToAdd,:,:,0] = example
        X_train[i,:,:,:,:] = output
        y_train.append(np.sum(exampleY))

    y_train     = np.array(y_train)
    if classify_or_regress == 'classify':
        y_train = np.equal.outer(y_train, np.arange(maxToAdd * 9)).astype(np.float)
    return X_train, y_train
