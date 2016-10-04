import numpy as np
from glob import glob
from scipy import misc
import re
from tqdm import tqdm
import data_loader

def evaluate_model(session,saver,pred,dropout_prob,checkpoint_ptr,data_mu,data_std):
    if saver == '':
        pass #session is loaded already
    else:
        saver.restore(session,checkpoint_ptr)#Load trained weights into model
    #Load data
    X_train_raw,y_train_temp,X_test_raw,y_test_temp,train_num,im_size = data_loader.train('test_multi_mnist')
    X_train_raw = X_train_raw.astype('float32')
    X_train_raw-=data_mu
    X_train_raw/=data_std #these parameters should come from the training data...
    data_size = X_train_raw.shape[0]
    cv_folds = data_size // s.batch_size
    x,y = prepare_mnist_data.repeat_adding_task(X_train_raw,y_train_temp,s.num_steps,data_size,[s.height,s.width],'regress') #turn regress into a variable passed from main
    cv_ind = range(data_size)
    #np.random.shuffle(cv_ind) #Dont need a shuffle for testing
    cv_ind = np.reshape(cv_ind,[cv_folds,s.batch_size])
    if dropout_prob == '':
        run_fun = lambda x: session.run(pred,feed_dict={X:x})    
    else:
        run_fun = lambda x: session.run(pred,feed_dict={X:x,keep_prob:dropout_prob})
    preds = np.zeros((x.shape[0]))
    diffs = np.zeros((x.shape[0]))
    for idx in range(cv_folds):
        test_idx = cv_ind[idx,:]
        bx = x[test_idx,:,:,:,:]
        by = y[test_idx]
        yhat = run_fun(bx)
	preds[test_idx] = yhat
        diffs[test_idx] = by - yhat
    return np.sum(diffs**2), preds    
