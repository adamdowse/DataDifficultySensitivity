

#Set of functions that are used in the main model file

import time
import tensorflow as tf
import numpy as np



def calc_train_loss_spectrum(dataset,model,limit=None):
    #Return the loss spectrum of all of the provided data.
    #dataset class
    #model class
    print('Calculating Loss Spectrum')
    t = time.time()
    dataset.build_train_iter() # build the iterator
    if limit != None:
        count = limit
    else:
        count = dataset.train_count

    loss_spectrum = np.zeros((count,1))
    for i in range(count//dataset.current_train_batch_size):
        imgs,labels = dataset.get_batch()
        loss_spectrum[i*dataset.current_train_batch_size:(i+1)*dataset.current_train_batch_size-1] = model.get_batch_loss(imgs,labels)
    print('--> time: ',time.time()-t)
    return loss_spectrum

def calc_S(ds,model,limit=None):
    #This calcualtes the S matrix trace with monty carlo estimation.
    # s_ij = E[softmax_i - softmax_j^2]
    print('Calculating S matrix trace')
    t = time.time()
    S = 0
    ds.build_train_iter()
    if limit != None and limit < ds.train_count:
        count = limit
    else:
        count = ds.train_count
    for i in range(count//ds.current_train_batch_size):
        imgs,labels = ds.get_batch()
        softmax_out = model.get_softmax(imgs)
        for i in range(ds.current_train_batch_size):
            for j in range(ds.num_classes):
                S += (softmax_out[j] - softmax_out[j])**2
    print('--> time: ',time.time()-t)
    return S/count

def calc_residuals(ds,model,limit=None):
    #This calcualtes the residuals of the model on the dataset
    #residuals = softmax - onehot
    print('Calculating Residuals')
    t = time.time()
    residuals = np.zeros((ds.num_classes,1))
    c = 0
    ds.build_train_iter(bs=ds.current_train_batch_size)
    if limit == None:
        count = ds.train_count
        print("Residuals limit not specified, using ",ds.train_count," data points")
    for i in range(count//ds.current_train_batch_size):
        imgs,labels = ds.get_batch()
        r = model.get_residuals(imgs,labels) # returns [bs x num_classes]
        residuals += tf.reduce_sum(r,axis=0)
        c += ds.current_train_batch_size
    residuals /= c
    print('--> time: ',time.time()-t)
    return residuals

def calc_d2zdw2(ds,model,limit=None):
    #calcualtes the second derivative of the model per softmax with respect to the weights
    print('Calculating d2zdw2')
    t = time.time()
    if limit == None:
        limit = ds.train_count
        print("d2zdw2 limit not specified, using ",ds.train_count," data points")
    for i in range(limit//ds.current_train_batch_size):
        items = ds.get_batch()
        d2zdw2 = model.Get_H(items)
        print([d.shape for d in d2zdw2])
        pnt()

def calc_R(ds,model,limit=None):
    #calcs trace of the NME matrix (the residual part of gauss-newton)
    print('Calculating R term')
    t = time.time()
    mean = 0
    count = 0
    if limit == None:
        limit = ds.train_count
        print("NME limit not specified, using ",ds.train_count," data points")
    else:
        print('limit: ',limit," data points")
    for i in range(limit//ds.current_train_batch_size):
        if i % 100 == 0:
            print(i)
        items = ds.get_batch()
        R = model.Get_R(items)
        count += 1
        delta = R - mean
        mean += delta/count
    print('--> time: ',time.time()-t)
    return mean
        
        

def calc_G(ds,model,limit=None):
    #calcs trace of the G matrix (the non residual part of gauss-newton)
    print('Calculating G')
    t = time.time()
    G_mean = 0
    S_mean = 0
    dzdt2_mean = 0
    count = 0

    if limit == None:
        limit = ds.train_count
        print("G limit not specified, using ",ds.train_count," data points")
    print('limit: ',limit//ds.current_train_batch_size)
    for i in range(limit//ds.current_train_batch_size):
        if i % 100 == 0:
            print(i)
        items = ds.get_batch()
        metrics = model.Get_G(items)
        
        count += 1
        deltaG = metrics[0] - G_mean
        G_mean += deltaG/count

        deltaS = metrics[1] - S_mean
        S_mean += deltaS/count

        deltaD = metrics[2] - dzdt2_mean
        dzdt2_mean += deltaD/count
    
    print('--> time: ',time.time()-t)
    return [G_mean,S_mean,dzdt2_mean]


        


def calc_dist_FIM(ds,model,FIM_bs,limit=None):
    #this needs to define the FIM
    #calc fim diag
    print('Calculating FIM (Distributed)')
    t = time.time()
    if limit == None:
        limit = ds.train_count
        print("FIM limit not specified, using ",ds.train_count," data points")

    replica_count = model.strategy.num_replicas_in_sync
    FIM_bs = FIM_bs//replica_count
    data_count = 0
    s = 0
    ds.build_train_iter(shuffle=True,bs=FIM_bs)
    for _ in range(limit//FIM_bs):
        if data_count/FIM_BS % 100 == 0:
            print(data_count)
        z = model.distributed_FIM_step(next(ds.iter_train))#send a batch to each replica, returns [replica_count x 1]
        s += tf.reduce_sum(z)
        data_count += len(z)
    mean = s/data_count
    print('--> time: ',time.time()-t)
    return mean

def calc_FIM(ds,model,FIM_bs,limit=None):
    print('Calculating FIM (Non-Distributed)')
    t = time.time()
    if limit == None:
        limit = ds.train_count
        print("FIM limit not specified, using ",ds.train_count," data points")
    
    data_count = 0
    s = 0
    ds.build_train_iter(shuffle=True,bs=FIM_bs)
    for _ in range(limit//FIM_bs):
        if data_count/FIM_bs % 100 == 0:
            print(data_count)
        z = model.Get_Z(ds.get_batch())#returns [FIM_bs x 1]
        s += tf.reduce_sum(z)
        data_count += FIM_bs
    mean = s/data_count
    print('--> time: ',time.time()-t)
    #convert to numpy
    mean = mean.numpy()
    return mean

def calc_FIM_var(ds,model,FIM_bs,limit=None):
    print('Calculating FIM (Non-Distributed) and Variance')
    t = time.time()
    if limit == None:
        limit = ds.train_count
        print("FIM limit not specified, using ",ds.train_count," data points")
    
    ds.build_train_iter(shuffle=True,bs=FIM_bs)
    #calc var as well as mean
    data_count = 0
    mean = 0
    var = 0
    for _ in range(lower_lim):
        if data_count % 500 == 0:
            print(data_count)
        x = model.Get_Z(next(ds.iter_train))#returns [FIM_bs x 1]
        for x in x:
            data_count += 1
            delta = x - mean 
            mean += delta/(data_count)
            var += delta*(x-mean)
        
    var /= data_count
    print('--> time: ',time.time()-t)
    return [mean,var]
   
