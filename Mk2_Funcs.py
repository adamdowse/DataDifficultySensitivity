

#Set of functions that are used in the main model file

import time
import tensorflow as tf
import numpy as np
import wandb
import csv

datetime = time.strftime("%Y%m%d-%H%M%S")

def calc_train_loss_spectrum(dataset,model,loss_func,limit=None,sort=True,save=False,groups=10):
    #Return the loss spectrum of all of the provided data.
    #dataset class
    #model class
    print('Calculating Loss Spectrum')
    t = time.time()
    if limit != None:
        count = limit
    else:
        count = dataset.train_count

    for step, (imgs, labels) in enumerate(dataset):
        if step % 100 == 0:
            print(step)
        with tf.GradientTape() as tape:
            losses = loss_func(labels, model(imgs))
        if step == 0:
            loss_spectrum = losses.numpy()
        else:
            loss_spectrum = np.append(loss_spectrum,losses.numpy())
       
    
    #sort lowest to highest
    loss_spectrum = np.sort(loss_spectrum)
    #create the stats from the loss spectrum so dont need to save the whole thing
    upper_bounds = np.zeros(groups)
    group_means = np.zeros(groups)
    group_medians = np.zeros(groups)
    for i in range(groups):
        #get the upper bound of each group
        upper_bounds[i] = loss_spectrum[int((i+1)*len(loss_spectrum)/groups)-1]
        #get the mean of each group
        group_means[i] = np.mean(loss_spectrum[int((i)*len(loss_spectrum)/groups):int((i+1)*len(loss_spectrum)/groups)])
        group_medians[i] = np.median(loss_spectrum[int((i)*len(loss_spectrum)/groups):int((i+1)*len(loss_spectrum)/groups)])
    #lowest loss
    lowest_bound = loss_spectrum[0]

    if save:
        combined_row = np.append(lowest_bound,upper_bounds,group_means)
        with open("LossSpectrums/"+datetime+wandb.run.id+".csv","a+") as f:
            writer = csv.writer(f)
            #write loss spectrum to file
            writer.writerow(combined_row)

    print('--> time: ',time.time()-t)
    return loss_spectrum,lowest_bound,upper_bounds,group_means,group_medians

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

def calc_FIM(ds,model,FIM_bs,bs,loss_func,limit=None,model_output_type='logit',groups=None):
    #model_output_type: 'logit' or 'softmax'
    print('Calculating FIM (Non-Distributed)')
    t = time.time()
    #calcualte the losses
    if groups == None:
        groups = 1
 
    losses,lowest_bound,upper_bounds,loss_group_means,loss_group_medians = calc_train_loss_spectrum(ds,model,loss_func,limit=limit,sort=True,save=False,groups=groups)

    if limit == None:
        limit = ds.train_count
        print("FIM limit not specified, using ",ds.train_count," data points")
    group_counts = np.zeros(groups)
    group_totals = np.zeros(groups)

    if FIM_bs != None:
        ds = ds.unbatch().batch(FIM_bs)

    def update_group_totals(loss,z):
        for (j_loss,j_z) in zip(loss.numpy(),z.numpy()):
            for g in range(groups):
                if j_loss <= upper_bounds[g]:
                    if group_counts[g] < limit:
                        group_counts[g] += 1
                        group_totals[g] += j_z
                        break
                    
        #if all groups have reached their limit
        if np.all(group_counts >= limit):
            return True
        else:
            return False

    for step, (imgs, labels) in enumerate(ds):
        if step % 100 == 0:
            print(step)
            print(group_counts)
        if model_output_type == 'logit':
            z,loss = model.Get_Z_logit((imgs,labels))
        elif model_output_type == 'softmax':
            z,loss = model.Get_Z_softmax((imgs,labels))

        maxed_groups = update_group_totals(loss,z)
        if maxed_groups:
            break
    #mean of each group
    group_means = group_totals/group_counts
    mean = np.mean(group_means)

    group_means = np.array(group_means)
    if FIM_bs != None:
        ds = ds.unbatch().batch(bs)
    print('--> time: ',time.time()-t)

    return group_means,mean,losses,lowest_bound,upper_bounds,loss_group_means,loss_group_medians


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

def calc_grad_alignment(ds,model,grad_bs,bs,limit=None,groups=None,upper_bounds=None):
    #calc the alignment of each subset to the average of all data
    print('Calculating Gradient Alignment')
    t = time.time()
    if limit == None:
        limit = ds.train_count
        print("Grad Alignment limit not specified, using ",ds.train_count," data points")
    if groups == None:
        groups = 1

    if upper_bounds is None:
        #need to calc the upper bounds
        _,_,upper_bounds,_,_ = calc_train_loss_spectrum(ds,model,tf.keras.losses.CategoricalCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE),limit=limit,sort=True,save=False,groups=groups)

    avg_grad = tf.zeros(model.get_params_shape())
    group_counts = np.zeros(groups)
    group_totals = np.zeros((groups,model.get_params_shape()))
    
    ds = ds.unbatch().batch(grad_bs)
    for step, batch in enumerate(ds):
        if step % 100 == 0:
            print(step)

        grads,losses = model.Get_grads(batch) 
        avg_grad += np.sum(grads,axis=0)
        for (j_grad,j_loss) in zip(grads,losses):
            for g in range(groups):
                if j_loss <= upper_bounds[g]:
                    if group_counts[g] < limit:
                        group_counts[g] += 1
                        group_totals[g] += j_grad
                        break
                    else:
                        continue
        if np.all(group_counts >= limit):
            break

    avg_grad /= np.sum(group_counts,axis=0) #average gradient
    #calc the average gradient for each group (need to divide by the number of data points in each group)
    for i in range(groups):
        group_totals[i] /= group_counts[i]

    #calc the norms
    avg_norm = np.linalg.norm(avg_grad)
    group_norms = np.linalg.norm(group_totals,axis=1) #check axis
    print('avg_norm: ',avg_norm)
    print('group_norms: ',group_norms)

    #calc the cosine similarity
    avg_cos = np.zeros(groups)
    for i in range(groups):
        if avg_norm*group_norms[i] == 0:
            avg_cos[i] = np.nan
        avg_cos[i] = np.dot(avg_grad,group_totals[i])/(avg_norm*group_norms[i])
    print('avg_cos: ',avg_cos)

    ds = ds.unbatch().batch(bs)
    print('--> time: ',time.time()-t)
    return avg_cos,avg_norm,group_norms

   

class CustomEOE(tf.keras.callbacks.Callback):
    #custom end of epoch callback
    def __init__(self,ds,model,config,loss_func):
        super().__init__()
        self.config = config
        self.loss_func = loss_func
        self.ds = ds
        self.model = model
    def on_epoch_end(self, epoch, logs=None):

        if epoch % self.config['epoch_calc_freq'] == 0:
            #calc FIM and or loss spectrum
            if self.config['FIM_calc'] == True and self.config['Loss_spec_calc'] == True:
                group_FIMs, mean_FIM,loss_spectrum,lowest_bound,upper_bounds,loss_group_means,loss_group_medians = calc_FIM(self.ds,
                    self.model,
                    self.config['FIM_bs'],
                    self.config['batch_size'],
                    self.loss_func,
                    limit=self.config['FIM_limit'],
                    model_output_type='softmax',
                    groups=self.config['FIM_groups'])
                logFIM = True
                logLoss = True
                
            elif self.config['FIM_calc'] == True:
                group_FIMs, mean_FIM,loss_spectrum,lowest_bound,upper_bounds,loss_group_means,loss_group_medians = calc_FIM(self.ds,
                    self.model,
                    self.config['FIM_bs'],
                    self.config['batch_size'],
                    self.loss_func,
                    limit=self.config['FIM_limit'],
                    model_output_type='softmax',
                    groups=self.config['FIM_groups'])
                logFIM = True
                logLoss = False

            elif self.config['Loss_spec_calc'] == True:
                loss_spectrum,lowest_bound,upper_bounds,loss_group_means,loss_group_medians = calc_train_loss_spectrum(self.ds,self.model,self.loss_func,limit=None,sort=True,save=True)
                logFIM = False
                logLoss = True
            
            else:
                logFIM = False
                logLoss = False

            #calc grad alignment
            if self.config['Grad_alignment_calc'] == True:
                if logLoss:
                    avg_cos,avg_norm,group_norms = calc_grad_alignment(self.ds,self.model,self.config['Grad_bs'],self.config['batch_size'],limit=self.config['FIM_limit'],groups=self.config['FIM_groups'],upper_bounds=upper_bounds)
                else:
                    avg_cos,avg_norm,group_norms = calc_grad_alignment(self.ds,self.model,self.config['Grad_bs'],self.config['batch_size'],limit=self.config['FIM_limit'],groups=self.config['FIM_groups'])
                wandb.log({'avg_norm':avg_norm},step=epoch)
                for i in range(len(group_norms)):
                    wandb.log({'group_norm_'+str(i):group_norms[i]},step=epoch)
                    wandb.log({'avg_cos_'+str(i):avg_cos[i]},step=epoch)

            #log the results
            if logFIM:
                for i in range(len(group_FIMs)):
                    wandb.log({'FIM_group_'+str(i):group_FIMs[i]},step=epoch)
                wandb.log({'FIM_mean':mean_FIM},step=epoch)
            if logLoss:
                wandb.log({'loss_spectrum':loss_spectrum},step=epoch)
                for i in range(len(upper_bounds)):
                    wandb.log({'lossUB_'+str(i):upper_bounds[i]},step=epoch)
                for i in range(len(loss_group_means)):
                    wandb.log({'loss_mean_'+str(i):loss_group_means[i]},step=epoch)
                for i in range(len(loss_group_medians)):
                    wandb.log({'loss_median_'+str(i):loss_group_medians[i]},step=epoch)
                wandb.log({'lossLowest':lowest_bound},step=epoch)

class CustomEOB(tf.keras.callbacks.Callback):
    #custom end of batch callback
    def __init__(self,ds,model,config,loss_func):
        super().__init__()
        self.config = config
        self.loss_func = loss_func
        self.ds = ds
        self.model = model
        self.epoch = 0
        self.num_batches = 0
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0:
            self.num_batches = self.latest_batch
    def on_batch_end(self, batch, logs=None):
        self.latest_batch = batch
        

        #Run FIM calculation and or loss spectrum calculation
        if batch % self.config['batch_calc_freq'] == 0 and self.epoch < self.config['batch_calc_epoch_limit']:
            if self.config['FIM_calc'] == True and self.config['Loss_spec_calc'] == True:
                #calc FIM and loss spectrum
                group_FIMs, mean_FIM,loss_spectrum,lowest_bound,upper_bounds,loss_group_means,loss_group_medians = calc_FIM(self.ds,
                    self.model,
                    self.config['FIM_bs'],
                    self.config['batch_size'],
                    self.loss_func,
                    limit=self.config['FIM_limit'],
                    model_output_type='softmax',
                    groups=self.config['FIM_groups'])
                logFIM = True
                logLoss = True

            elif self.config['FIM_calc'] == True:
                #calc just the FIM
                group_FIMs, mean_FIM,loss_spectrum,lowest_bound,upper_bounds,loss_group_means,loss_group_medians = calc_FIM(self.ds,
                    self.model,
                    self.config['FIM_bs'],
                    self.config['batch_size'],
                    self.loss_func,
                    limit=self.config['FIM_limit'],
                    model_output_type='softmax',
                    groups=self.config['FIM_groups'])
                logFIM = True
                logLoss = False

            elif self.config['Loss_spec_calc'] == True:
                #calc just the loss spectrum
                loss_spectrum,lowest_bound,upper_bounds,loss_group_means,loss_group_medians = calc_train_loss_spectrum(self.ds,self.model,self.loss_func,limit=None,sort=True,save=True)
                logFIM = False
                logLoss = True
            
            else:
                logFIM = False
                logLoss = False

            #calc grad alignment
            if self.config['Grad_alignment_calc'] == True:
                if logLoss:
                    avg_cos,avg_norm,group_norms = calc_grad_alignment(self.ds,self.model,self.config['Grad_bs'],self.config['batch_size'],limit=self.config['FIM_limit'],groups=self.config['FIM_groups'],upper_bounds=upper_bounds)
                else:
                    avg_cos,avg_norm,group_norms = calc_grad_alignment(self.ds,self.model,self.config['Grad_bs'],self.config['batch_size'],limit=self.config['FIM_limit'],groups=self.config['FIM_groups'])
                print('avg_norm: ',avg_norm)
                print('group_norms: ',len(group_norms))
                print('avg_cos: ',len(avg_cos))
                wandb.log({'b_avg_norm':avg_norm},step=(self.epoch*self.num_batches)+batch)
                for i in range(len(group_norms)):
                    wandb.log({'b_group_norm_'+str(i):group_norms[i]},step=(self.epoch*self.num_batches)+batch)
                    wandb.log({'b_avg_cos_'+str(i):avg_cos[i]},step=(self.epoch*self.num_batches)+batch)

            #log the results
            if logFIM:
                for i in range(len(group_FIMs)):
                    wandb.log({'b_FIM_group_'+str(i):group_FIMs[i]},step=(self.epoch*self.num_batches)+batch)
                wandb.log({'b_FIM_mean':mean_FIM},step=(self.epoch*self.num_batches)+batch)
            if logLoss:
                wandb.log({'b_loss_spectrum':loss_spectrum},step=(self.epoch*self.num_batches)+batch)
                for i in range(len(upper_bounds)):
                    wandb.log({'b_lossUB_'+str(i):upper_bounds[i]},step=(self.epoch*self.num_batches)+batch)
                for i in range(len(loss_group_means)):
                    wandb.log({'b_loss_mean_'+str(i):loss_group_means[i]},step=(self.epoch*self.num_batches)+batch)
                for i in range(len(loss_group_medians)):
                    wandb.log({'b_loss_median_'+str(i):loss_group_medians[i]},step=(self.epoch*self.num_batches)+batch)
                wandb.log({'b_lossLowest':lowest_bound},step=(self.epoch*self.num_batches)+batch)
            


            