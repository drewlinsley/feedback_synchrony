import numpy as np
from glob import glob
from scipy import misc
import re
from tqdm import tqdm
from ops import data_loader
from ops.utils import * #_variable_with_weight_decay
from ops import model

def evaluate_model(meta_data,session,saver,x,y,checkpoint_ptr):
    #Load model metadata
    holder = np.load(meta_data)
    settings = holder['settings'].item(0)
    s = Map(settings)
    task = parse_task(s)

    #Load model weights
    if saver != '' or session != '':
        pass #session is loaded already
    else:
        session, init_vars, merged, saver, optim, writer, cost, keep_prob, X, targets, Wc, Uc, Wg, Ug, c, state, pred, accuracy = model.build_model(s)
        session.run(init_vars)
    saver.restore(session,checkpoint_ptr)#Load trained weights into model

    #Load data
    if x == '' or y == '':
        X_train_raw,y_train_temp,X_test_raw,y_test_temp,_,im_size,num_channels,cats = data_loader.train(which_data=s.which_data,num_steps=s.num_steps)
        X_train_raw,data_mu,data_std = data_loader.normalize(X_train_raw,zm=True,uv=True)
        data_size = X_test_raw.shape[0]
        x,y = task(X_test_raw,y_test_temp,s.num_steps,data_size,[s.height,s.width,s.channels],s.output_shape) #turn regress into a variable passed from main
    else:
        data_size = X_test_raw.shape[0] #using previously loaded data

    if s.dropout_prob == '':
        run_fun = lambda x: session.run(pred,feed_dict={X:x})    
    else:
        run_fun = lambda x: session.run(pred,feed_dict={X:x,keep_prob:1})
    cv_folds = data_size // s.batch_size
    cv_ind = range(data_size)
    #np.random.shuffle(cv_ind)
    cv_ind = cv_ind[:(cv_folds*s.batch_size)]
    cv_ind = np.reshape(cv_ind,[cv_folds,s.batch_size])
    yhat = np.zeros((y.shape[0],1))
    for idx in tqdm(range(cv_folds)):
        cv_idx = cv_ind[idx,:]
        bx = x[cv_idx,:,:,:,:]
        by = y[cv_idx]    
        it_yhat = run_fun(bx)
        yhat[cv_idx] = np.argmax(it_yhat,axis=1).reshape(len(cv_idx),1)
    if s.output_shape > 1:
        summary_stat = np.mean((yhat==y).astype(np.float32))
    else:
        summary_stat = np.sum(diffs**2)    
    return summary_stat, yhat, x, y, session, saver
