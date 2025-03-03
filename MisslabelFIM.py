#misslabel percentage of the cifar10 dfataset and record the FIM of each subset over training

import numpy as np
import tensorflow as tf

import lr_schedules

from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import wandb 
import os
from scipy.optimize import curve_fit

class AugmentLossFIMCallback(keras.callbacks.Callback):
    #Show data that is correctly classified against not correctly classified
    def __init__(self, ds, epochRecord,loss_func, limit=500,doNorm=False,prefix="",do_batch=False):
        self.ds = ds
        self.epoch = 0
        self.epochRecord = epochRecord
        self.limit = limit
        self.df = pd.DataFrame(columns=['id'])
        self.loss_function = loss_func
        self.doNorm = doNorm
        self.prefix = prefix
        self.do_batch = do_batch
        self.idx = 0

    @tf.function
    def Get_Z_sm(self,items):
        #take just one image and augment it n times recording the FIM
        x,y = items #this is a single image and label
        
        x = tf.expand_dims(x,0)
        y = tf.expand_dims(y,0)

        bs = tf.shape(x)[0]
        
        #x = tf.image.random_flip_left_right(x)
        #x = tf.image.random_rotation(x,0.5)
        r = tf.random.uniform((),0,1)
        x = x + tf.cast(tf.random.normal(x.shape,0,r),tf.float64)
        
        with tf.GradientTape() as tape:
            y_hat = self.model(x,training=False) #get model output (softmax) [BS x num_classes]
            loss = self.loss_function(y,y_hat)
            selected = tf.squeeze(tf.random.categorical(tf.math.log(y_hat), 1)) #sample from the output [BS x 1]
            output = tf.gather(y_hat,selected,axis=1) #get the output for the selected class [BS x 1]
            output = tf.math.log(output)#tf.math.log(output) #log the output [BS x 10]
        
        j = tape.jacobian(output,self.model.trainable_variables)
        layer_sizes = [tf.reduce_sum(tf.size(v)) for v in self.model.trainable_variables] #get the size of each layer
        j = [tf.reshape(j[i],(bs,layer_sizes[i])) for i in range(len(j))] #reshape the gradient to [10 x num_layer_params x layers]
        j = tf.concat(j,axis=1) #concat the gradient over the layers [10 x num_params]
        j = tf.square(j) #square the gradient [10 x num_params]
        j = tf.reduce_sum(j,axis=1) #sum the gradient [ 10]
        if self.doNorm:
            j = j/tf.cast(tf.reduce_sum(layer_sizes),tf.float32)
        r = tf.expand_dims(r,0)
        return j, loss,y_hat,y,r



    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch in self.epochRecord) and not self.do_batch:
            print("doing LossFIMRecord")
            c = 0
            items = next(iter(self.ds))
            #get first image and label in the batch
            items = (items[0][0],items[1][0])
            print(items[0].shape)
            print(items[1].shape)

            for i in range(self.limit):
                j, loss,y_hat,y,r = self.Get_Z_sm(items)
                if c == 0:
                    #where the model is correct
                    FIMs = j
                    Losses = loss
                    R = r
                    c += 1
                else:
                    FIMs = tf.concat([FIMs,j],axis=0)
                    Losses = tf.concat([Losses,loss],axis=0)
                    R = tf.concat([R,r],axis=0)
                    c += 1
            
            FIMs = tf.squeeze(FIMs)
            Losses = tf.squeeze(Losses)
            R = tf.squeeze(R)
            
            #add the FIMs to the dataframe
            self.df = pd.concat([self.df,pd.DataFrame({str(epoch)+"FIM":FIMs.numpy(),str(epoch)+"Loss":Losses.numpy(),str(epoch)+"R":R.numpy()})],ignore_index=True,axis=1)
            
            #self.df = self.df.append({str(self.epoch)+"FIM":FIMs[i].numpy(),str(self.epoch)+"Loss":Losses[i].numpy()},ignore_index=True)
    
    def on_train_end(self, logs=None):
        self.df.to_csv(str(self.prefix)+"LossFIM.csv")

class LayerLossFIMCallback(keras.callbacks.Callback):
    #Show The relationship between Loss and the FIM of each layer
    def __init__(self, ds, epochRecord,loss_func, limit=500,doNorm=False,prefix="",do_batch=False):
        self.ds = ds
        self.epoch = 0
        self.epochRecord = epochRecord
        self.limit = limit
        self.df = pd.DataFrame(columns=['id'])
        self.loss_function = loss_func
        self.doNorm = doNorm
        self.prefix = prefix
        self.do_batch = do_batch

    @tf.function
    def Get_Z_sm(self,items):
        x,y = items
        bs = tf.shape(x)[0]
        
        with tf.GradientTape() as tape:
            #print(text_batch[i])
            y_hat = self.model(x,training=False) #get model output (softmax) [BS x num_classes]
            loss = self.loss_function(y,y_hat)
            #Selection based of prop output distribution
            selected = tf.squeeze(tf.random.categorical(tf.math.log(y_hat), 1)) #sample from the output [BS x 1]
            #uniform random selection
            #selected = tf.squeeze(tf.random.uniform((bs,),0,10,dtype=tf.int64))
            output = tf.gather(y_hat,selected,axis=1,batch_dims=1) #get the output for the selected class [BS x 1]
            output = tf.math.log(output)#tf.math.log(output) #log the output [BS x 1]

        j = tape.jacobian(output,self.model.trainable_variables)
        layer_sizes = [tf.reduce_sum(tf.size(v)) for v in self.model.trainable_variables] #get the size of each layer

        j = [tf.reshape(j[i],(bs,layer_sizes[i])) for i in range(len(j))] #reshape the gradient to [BS x num_layer_params x layers]
        #square the gradient [BS x num_params]
        j = [tf.square(j[i]) for i in range(len(j))] #square the gradient [BS x num_layer_params x layers]
        #sum the values of each layer so size is [BS x num_layers]
        j = [tf.reduce_sum(j[i],axis=1) for i in range(len(j))] #sum the gradient [BS x num_layers]

        return j, loss,output,y



    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch in self.epochRecord) and not self.do_batch:
            print("doing LossFIMRecord")
            c = 0
            for items in self.ds:
                if c*items[0].shape[0] > self.limit:
                    break
                j, loss,y_hat,y = self.Get_Z_sm(items) #j= [BS x num_layers] loss = [BS x 1] y_hat = [BS x num_classes] y = [BS x num_classes]
                
                if c == 0:
                    FIMs = j
                    Losses = loss
                    y_hats = y_hat
                    c += 1
                else:
                    #correct = tf.concat([correct,tf.argmax(y_hat,axis=1) == tf.argmax(y,axis=1)],axis=0)
                    FIMs = tf.concat([FIMs,j],axis=1)
                    Losses = tf.concat([Losses,loss],axis=0)
                    y_hats = tf.concat([y_hats,y_hat],axis=0)
                    c += 1
            
            FIMs = tf.squeeze(FIMs)
            Losses = tf.squeeze(Losses)
            y_hats = tf.squeeze(y_hats)

            def func(x,a):
                return a*np.log(1+x)**2

            #add the FIMs to the dataframe
            alpha = []
            for l in range(len(FIMs)):
                #calc the alpha for the epoch
                popt, pcov = curve_fit(func, -y_hats.numpy(),FIMs[l].numpy())
                alpha.append(popt[0])

            self.df = pd.concat([self.df,pd.DataFrame({str(epoch)+"alpha":alpha})],ignore_index=True,axis=1)
            


    def on_train_end(self, logs=None):
        self.df.to_csv(str(self.prefix)+"LossFIM.csv")

class CorrectLossFIMCallback(keras.callbacks.Callback):
    #Show data that is correctly classified against not correctly classified
    def __init__(self, ds, epochRecord,loss_func, limit=500,doNorm=False,prefix="",do_batch=False):
        self.ds = ds
        self.epoch = 0
        self.epochRecord = epochRecord
        self.limit = limit
        self.df = pd.DataFrame(columns=['id'])
        self.loss_function = loss_func
        self.doNorm = doNorm
        self.prefix = prefix
        self.do_batch = do_batch

    @tf.function
    def Get_Z_sm(self,items):
        x,y = items
        bs = tf.shape(x)[0]
        
        with tf.GradientTape() as tape:
            #print(text_batch[i])
            y_hat = self.model(x,training=False) #get model output (softmax) [BS x num_classes]
            loss = self.loss_function(y,y_hat)
            selected = tf.squeeze(tf.random.categorical(tf.math.log(y_hat), 1)) #sample from the output [BS x 1]
            #uniform random selection
            #selected = tf.squeeze(tf.random.uniform((bs,),0,10,dtype=tf.int64))
            output = tf.gather(y_hat,selected,axis=1,batch_dims=1) #get the output for the selected class [BS x 1]

            #randomly create a values between 0 and 1 TODO THIS IS CAUSING A PROBLEM
            #output = tf.squeeze(tf.random.uniform((bs,),0.1,0.9,dtype=tf.float64))
            output = tf.math.log(output)#tf.math.log(output) #log the output [BS x 1]

        j = tape.jacobian(output,self.model.trainable_variables)
        #print(j)
        layer_sizes = [tf.reduce_sum(tf.size(v)) for v in self.model.trainable_variables] #get the size of each layer
        j = [tf.reshape(j[i],(bs,layer_sizes[i])) for i in range(len(j))] #reshape the gradient to [BS x num_layer_params x layers]
        j = tf.concat(j,axis=1) #concat the gradient over the layers [BS x num_params]
        j = tf.square(j) #square the gradient [BS x num_params]
        j = tf.reduce_sum(j,axis=1) #sum the gradient [ 1]
        if self.doNorm:
            j = j/tf.cast(tf.reduce_sum(layer_sizes),tf.float32)

        sel = tf.argmax(y,axis=1) == selected

        #-log(y_hat) is the log of the output
        #new_selected = tf.squeeze(tf.random.uniform((bs,),0,10,dtype=tf.int64))
        #new_output = tf.gather(y_hat,new_selected,axis=1,batch_dims=1) #get the output for the selected class [BS x 1]

        #randomly create a values between 0 and 1
        #new_output = tf.squeeze(tf.random.uniform((bs,),0,1,dtype=tf.float32))
        #new_output = tf.math.log(new_output)#tf.math.log(output) #log the output [BS x 1]

        #new_sel = selected == new_selected

        return j, loss,y_hat,y,sel,-output  #,-new_output,new_sel,-output



    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch in self.epochRecord) and not self.do_batch:
            print("doing LossFIMRecord")
            c = 0
            for items in self.ds:
                if c*items[0].shape[0] > self.limit:
                    break
                j, loss,y_hat,y,sel,o = self.Get_Z_sm(items)
                if c == 0:
                    #where the model is correct
                    #correct = tf.argmax(y_hat,axis=1) == tf.argmax(y,axis=1)
                    selected = sel
                    FIMs = j
                    Losses = loss
                    #new_selected = new_sel
                    #new_output = new_o
                    output = o
                    c += 1
                else:
                    #correct = tf.concat([correct,tf.argmax(y_hat,axis=1) == tf.argmax(y,axis=1)],axis=0)
                    FIMs = tf.concat([FIMs,j],axis=0)
                    Losses = tf.concat([Losses,loss],axis=0)
                    selected = tf.concat([selected,sel],axis=0)
                    #new_selected = tf.concat([new_selected,new_sel],axis=0)
                    #new_output = tf.concat([new_output,new_o],axis=0)
                    output = tf.concat([output,o],axis=0)
                    c += 1
            
            FIMs = tf.squeeze(FIMs)
            Losses = tf.squeeze(Losses)
            #correct = tf.squeeze(correct)
            selected = tf.squeeze(selected)
            #new_selected = tf.squeeze(new_selected)
            #new_output = tf.squeeze(new_output)
            output = tf.squeeze(output)

            def func(x,a):
                return  a*np.log(1+x)**2

            popt, pcov = curve_fit(func, output.numpy(), FIMs.numpy())
            print(popt)

            #add the FIMs to the dataframe
            #self.df = pd.concat([self.df,pd.DataFrame({str(epoch)+"FIM":FIMs.numpy(),str(epoch)+"Loss":Losses.numpy(),str(epoch)+"Selected":selected.numpy()})],ignore_index=True,axis=1)
            #self.df = pd.concat([self.df,pd.DataFrame({str(epoch)+"FIM":FIMs.numpy(),str(epoch)+"Selected":selected.numpy(),str(epoch)+"NewOutput":new_output.numpy(),str(epoch)+"NewSelected":new_selected.numpy()})],ignore_index=True,axis=1)
            #self.df = pd.concat([self.df,pd.DataFrame({str(self.epoch)+"FIM":FIMs.numpy(),str(self.epoch)+"Loss":Losses.numpy(),str(epoch)+"Selected":selected.numpy()})],ignore_index=True,axis=1)
            #self.df = pd.concat([self.df,pd.DataFrame({str(epoch)+"NewOutput":new_output.numpy(),str(epoch)+"Output":output.numpy(),str(epoch)+"NewSelected":new_selected.numpy()})],ignore_index=True,axis=1)
            #self.df = pd.concat([self.df,pd.DataFrame({str(epoch)+"FIM":FIMs.numpy(),str(epoch)+"Output":output.numpy()})],ignore_index=True,axis=1)
            self.df = pd.concat([self.df,pd.DataFrame({str(epoch)+"alpha":popt[0],str(epoch)+"a_var":pcov[0]})],ignore_index=True,axis=1)


    def on_train_end(self, logs=None):
        self.df.to_csv(str(self.prefix)+"LossFIM.csv")

class gradMagCallback(keras.callbacks.Callback):
    def __init__(self, ds, epochRecord,loss_func, limit=500,doNorm=False,prefix="",do_batch=False):
        self.ds = ds
        self.epoch = 0
        self.epochRecord = epochRecord
        self.limit = limit
        self.df = pd.DataFrame(columns=['id'])
        self.loss_function = loss_func
        self.doNorm = doNorm
        self.prefix = prefix
        self.do_batch = do_batch

    @tf.function
    def Get_Mag(self,items):
        x,y = items
        bs = tf.shape(x)[0]
        
        with tf.GradientTape() as tape:
            #print(text_batch[i])
            y_hat = self.model(x,training=False) #get model output (softmax) [BS x num_classes]
            loss = self.loss_function(y,y_hat)
            
        j = tape.jacobian(loss,self.model.trainable_variables)
        layer_sizes = [tf.reduce_sum(tf.size(v)) for v in self.model.trainable_variables]
        j = [tf.reshape(j[i],(bs,layer_sizes[i])) for i in range(len(j))]
        j = tf.concat(j,axis=1)
        j = tf.norm(j,axis=1)
        return j, loss

    def on_batch_end(self, batch, logs=None):
        if self.do_batch and (self.epoch in self.epochRecord):
            c = 0
            items = next(iter(self.ds))
            j, loss = self.Get_Mag(items)
            Mag = j
            Losses = loss
                    
            Mags = tf.squeeze(Mags)
            Losses = tf.squeeze(Losses)
            #add the FIMs to the dataframe
            self.df = pd.concat([self.df,pd.DataFrame({str(batch)+"Mag":Mags.numpy(),str(batch)+"Loss":Losses.numpy()})],ignore_index=True,axis=1)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        if self.do_batch and (self.epoch in self.epochRecord):
            print("doing LossMagRecord per batch")
        
    def on_epoch_end(self, epoch, logs=None):
        if self.do_batch and (self.epoch in self.epochRecord):
            self.df.to_csv(str(self.prefix)+str(epoch)+"LossMag.csv")
        if (epoch in self.epochRecord) and not self.do_batch:
            print("doing LossMagRecord")
            c = 0
            for items in self.ds:
                if c*items[0].shape[0] > self.limit:
                    break
                j, loss = self.Get_Mag(items)
                if c == 0:
                    Mags = j
                    Losses = loss
                    c += 1
                else:
                    Mags = tf.concat([Mags,j],axis=0)
                    Losses = tf.concat([Losses,loss],axis=0)
                    c += 1
            
            Mags = tf.squeeze(Mags)
            Losses = tf.squeeze(Losses)
            #add the FIMs to the dataframe
            self.df = pd.concat([self.df,pd.DataFrame({str(epoch)+"Mag":Mags.numpy(),str(epoch)+"Loss":Losses.numpy()})],ignore_index=True,axis=1)
            
            #self.df = self.df.append({str(self.epoch)+"FIM":FIMs[i].numpy(),str(self.epoch)+"Loss":Losses[i].numpy()},ignore_index=True)
    
    def on_train_end(self, logs=None):
        self.df.to_csv(str(self.prefix)+"LossMag.csv")

class LossFIMCallback(keras.callbacks.Callback):
    def __init__(self, ds, epochRecord,loss_func, limit=500,doNorm=False,prefix="",do_batch=False,do_Mag=False):
        self.ds = ds
        self.epoch = 0
        self.epochRecord = epochRecord
        self.limit = limit
        self.df = pd.DataFrame(columns=['id'])
        self.loss_function = loss_func
        self.doNorm = doNorm
        self.prefix = prefix
        self.do_batch = do_batch
        self.do_Mag = do_Mag

    @tf.function
    def Get_Mag(self,items):
        x,y = items
        bs = tf.shape(x)[0]
        
        with tf.GradientTape() as tape:
            #print(text_batch[i])
            y_hat = self.model(x,training=False) #get model output (softmax) [BS x num_classes]
            loss = self.loss_function(y,y_hat)
            
        j = tape.jacobian(loss,self.model.trainable_variables)
        layer_sizes = [tf.reduce_sum(tf.size(v)) for v in self.model.trainable_variables]
        j = [tf.reshape(j[i],(bs,layer_sizes[i])) for i in range(len(j))]
        j = tf.concat(j,axis=1)
        j = tf.norm(j,axis=1)
        return j

    @tf.function
    def Get_Z_sm(self,items):
        x,y = items
        bs = tf.shape(x)[0]
        
        with tf.GradientTape() as tape:
            #print(text_batch[i])
            y_hat = self.model(x,training=False) #get model output (softmax) [BS x num_classes]
            loss = self.loss_function(y,y_hat)
            selected = tf.squeeze(tf.random.categorical(tf.math.log(y_hat), 1)) #sample from the output [BS x 1]
            output = tf.gather(y_hat,selected,axis=1,batch_dims=1) #get the output for the selected class [BS x 1]
            output = tf.math.log(output)#tf.math.log(output) #log the output [BS x 1]

        j = tape.jacobian(output,self.model.trainable_variables)
        layer_sizes = [tf.reduce_sum(tf.size(v)) for v in self.model.trainable_variables] #get the size of each layer
        j = [tf.reshape(j[i],(bs,layer_sizes[i])) for i in range(len(j))] #reshape the gradient to [BS x num_layer_params x layers]
        j = tf.concat(j,axis=1) #concat the gradient over the layers [BS x num_params]
        j = tf.square(j) #square the gradient [BS x num_params]
        j = tf.reduce_sum(j,axis=1) #sum the gradient [ 1]
        if self.doNorm:
            j = j/tf.cast(tf.reduce_sum(layer_sizes),tf.float32)
        return j, loss

    def on_batch_end(self, batch, logs=None):
        if self.do_batch and (self.epoch in self.epochRecord):
            c = 0
            items = next(iter(self.ds))
            j, loss = self.Get_Z_sm(items)
            FIMs = j
            Losses = loss
                    
            FIMs = tf.squeeze(FIMs)
            Losses = tf.squeeze(Losses)
            #add the FIMs to the dataframe
            self.df = pd.concat([self.df,pd.DataFrame({str(batch)+"FIM":FIMs.numpy(),str(batch)+"Loss":Losses.numpy()})],ignore_index=True,axis=1)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        if self.do_batch and (self.epoch in self.epochRecord):
            print("doing LossFIMRecord per batch")
        
    def on_epoch_end(self, epoch, logs=None):
        if self.do_batch and (self.epoch in self.epochRecord):
            self.df.to_csv(str(self.prefix)+str(epoch)+"LossFIM.csv")
        if (epoch in self.epochRecord) and not self.do_batch:
            print("doing LossFIMRecord")
            c = 0
            for items in self.ds:
                if c*items[0].shape[0] > self.limit:
                    break
                j, loss = self.Get_Z_sm(items)
                if self.do_Mag:
                    mag = self.Get_Mag(items)
                if c == 0:
                    FIMs = j
                    Losses = loss
                    if self.do_Mag:
                        Mags = mag
                    c += 1
                else:
                    FIMs = tf.concat([FIMs,j],axis=0)
                    Losses = tf.concat([Losses,loss],axis=0)
                    if self.do_Mag:
                        Mags = tf.concat([Mags,mag],axis=0)
                    c += 1
            
            FIMs = tf.squeeze(FIMs)
            Losses = tf.squeeze(Losses)
            if self.do_Mag:
                Mags = tf.squeeze(Mags)
            #add the FIMs to the dataframe
            if self.do_Mag:
                self.df = pd.concat([self.df,pd.DataFrame({str(epoch)+"FIM":FIMs.numpy(),str(epoch)+"Loss":Losses.numpy(),str(epoch)+"Mag":Mags.numpy()})],ignore_index=True,axis=1)
            else:
                self.df = pd.concat([self.df,pd.DataFrame({str(epoch)+"FIM":FIMs.numpy(),str(epoch)+"Loss":Losses.numpy()})],ignore_index=True,axis=1)
            
            #self.df = self.df.append({str(self.epoch)+"FIM":FIMs[i].numpy(),str(self.epoch)+"Loss":Losses[i].numpy()},ignore_index=True)
    
    def on_train_end(self, logs=None):
        self.df.to_csv(str(self.prefix)+"LossFIM.csv")

class GFIMCallback(keras.callbacks.Callback):
    def __init__(self, ds, doStep,num_groups,loss_func,doNorm=False,name_prefix="",limit=500):
        self.doStep = doStep
        self.ds = ds
        self.num_groups = num_groups
        self.epoch = 0
        self.curr_batch = 0
        self.max_batch = 0
        self.loss_function = loss_func
        self.name_prefix = name_prefix
        self.doNorm = doNorm
        self.limit = limit

    @tf.function
    def Get_Z_sm(self,items):
        x,y = items
        bs = tf.shape(x)[0]
        
        with tf.GradientTape() as tape:
            #print(text_batch[i])
            y_hat = self.model(x,training=False) #get model output (softmax) [BS x num_classes]
            loss = self.loss_function(y,y_hat)
            selected = tf.squeeze(tf.random.categorical(tf.math.log(y_hat), 1)) #sample from the output [BS x 1]
            output = tf.gather(y_hat,selected,axis=1,batch_dims=1) #get the output for the selected class [BS x 1]
            output = tf.math.log(output)#tf.math.log(output) #log the output [BS x 1]

        j = tape.jacobian(output,self.model.trainable_variables)
        layer_sizes = [tf.reduce_sum(tf.size(v)) for v in self.model.trainable_variables] #get the size of each layer
        j = [tf.reshape(j[i],(bs,layer_sizes[i])) for i in range(len(j))] #reshape the gradient to [BS x num_layer_params x layers]
        j = tf.concat(j,axis=1) #concat the gradient over the layers [BS x num_params]
        j = tf.square(j) #square the gradient [BS x num_params]
        j = tf.reduce_sum(j,axis=1) #sum the gradient [ 1]
        if self.doNorm:
            j = j/tf.cast(tf.reduce_sum(layer_sizes),tf.float32)
        return j, loss

    def on_epoch_end(self, epoch, logs=None):
        print("doing "+str(self.name_prefix))
        if epoch == 0:
            self.max_batch = self.curr_batch
        self.epoch = epoch
        if not self.doStep:
            c = 0
            for items in self.ds:
                if c*items[0].shape[0] > self.limit:
                    break
                j, loss = self.Get_Z_sm(items)
                if c == 0:
                    FIMs = j
                    Losses = loss
                    c += 1
                else:
                    FIMs = tf.concat([FIMs,j],axis=0)
                    Losses = tf.concat([Losses,loss],axis=0)
                    c += 1
            
            FIMs = tf.squeeze(FIMs)
            Losses = tf.squeeze(Losses)
            #reorder the FIMs based on the losses
            idx = tf.argsort(Losses)
            group_count = tf.shape(FIMs)[0]//self.num_groups
            for i in range(self.num_groups):
                if i == self.num_groups-1:
                    group_idx = idx[i*group_count:]
                group_idx = idx[i*group_count:(i+1)*group_count]
                
                wandb.log({str(self.name_prefix)+"GFIM_"+str(i):tf.reduce_mean(tf.gather(FIMs,group_idx),axis=0)},step=epoch)
                wandb.log({str(self.name_prefix)+"Loss_"+str(i):tf.reduce_mean(tf.gather(Losses,group_idx),axis=0)},step=epoch)
            wandb.log({str(self.name_prefix)+"Loss":tf.reduce_mean(Losses,axis=0)},step=epoch)
            wandb.log({str(self.name_prefix)+"FIM":tf.reduce_mean(FIMs,axis=0)},step=epoch)

    def on_batch_end(self, batch, logs=None):
        if self.doStep:
            c = 0
            for items in self.ds:
                j, loss = self.Get_Z_sm(items)
                if c == 0:
                    FIMs = j
                    Losses = loss
                    c += 1
                else:
                    FIMs = tf.concat([FIMs,j],axis=0)
                    Losses = tf.concat([Losses,loss],axis=0)
            FIMs = tf.squeeze(FIMs)
            Losses = tf.squeeze(Losses)
            #reorder the FIMs based on the losses
            idx = tf.argsort(Losses)
            group_count = tf.shape(FIMs)[0]//self.num_groups
            for i in range(self.num_groups):
                if i == self.num_groups-1:
                    group_idx = idx[i*group_count:]
                group_idx = idx[i*group_count:(i+1)*group_count]
                wandb.log({str(self.name_prefix)+"GFIM_"+str(i):tf.reduce_mean(tf.gather(FIMs,group_idx),axis=0)},step=(self.epoch*self.max_batch)+self.curr_batch)
                wandb.log({str(self.name_prefix)+"Loss_"+str(i):tf.reduce_mean(tf.gather(Losses,group_idx),axis=0)},step=(self.epoch*self.max_batch)+self.curr_batch)
        self.curr_batch += 1

class Calc_K_On_End(keras.callbacks.Callback):
    def __init__(self, ds,loss_func, limit=500,save=False,prefix=""):
        self.ds = ds
        self.limit = limit
        self.df = pd.DataFrame(columns=['id'])
        self.loss_function = loss_func
        self.save = save
        self.prefix = prefix
        self.num_classes = 10
        self.filter_strength = 5
        self.K = 0
        self.K_conf = 0
        self.OG_acc_accum = []
        self.lowest_C_acc_accum = []

    @tf.function
    def Get_Z_sm(self,items):
        x,y = items
        bs = tf.shape(x)[0]
        
        with tf.GradientTape() as tape:
            #print(text_batch[i])
            y_hat = self.model(x,training=False) #get model output (softmax) [BS x num_classes]
            loss = self.loss_function(y,y_hat)
            selected = tf.squeeze(tf.random.categorical(tf.math.log(y_hat), 1)) #sample from the output [BS x 1]
            output = tf.gather(y_hat,selected,axis=1,batch_dims=1) #get the output for the selected class [BS x 1]
            output = tf.math.log(output)#tf.math.log(output) #log the output [BS x 1]

        j = tape.jacobian(output,self.model.trainable_variables)
        layer_sizes = [tf.reduce_sum(tf.size(v)) for v in self.model.trainable_variables] #get the size of each layer
        j = [tf.reshape(j[i],(bs,layer_sizes[i])) for i in range(len(j))] #reshape the gradient to [BS x num_layer_params x layers]
        j = tf.concat(j,axis=1) #concat the gradient over the layers [BS x num_params]
        j = tf.square(j) #square the gradient [BS x num_params]
        j = tf.reduce_sum(j,axis=1) #sum the gradient [ 1]
        return j, loss

    def get_bs(self,array):
        return tf.shape(array)[0]

    @tf.function
    def Get_Z_sm_class(self,imgs,class_idx):
        bs = self.get_bs(imgs)
        print(class_idx)
        print(class_idx.shape)
        
        with tf.GradientTape() as tape:
            y_hat = self.model(imgs,training=False) #get model output (softmax) [BS x num_classes]
            output = tf.gather(y_hat,class_idx,axis=1,batch_dims=1) #get the output for the selected class [BS x 1]
            output = tf.math.log(output)#tf.math.log(output) #log the output [BS x 1]

        j = tape.jacobian(output,self.model.trainable_variables)
        layer_sizes = [tf.reduce_sum(tf.size(v)) for v in self.model.trainable_variables] #get the size of each layer
        j = [tf.reshape(j[i],(bs,layer_sizes[i])) for i in range(len(j))] #reshape the gradient to [BS x num_layer_params x layers]
        j = tf.concat(j,axis=1) #concat the gradient over the layers [BS x num_params]
        j = tf.square(j) #square the gradient [BS x num_params]
        j = tf.reduce_sum(j,axis=1) #sum the gradient [ 1]
        return j

    def on_train_end(self, logs=None):
        c = 0
        for items in self.ds:
            if c*items[0].shape[0] > self.limit:
                break
            j, loss = self.Get_Z_sm(items)
            if c == 0:
                FIMs = j
                Losses = loss
                c += 1
            else:
                FIMs = tf.concat([FIMs,j],axis=0)
                Losses = tf.concat([Losses,loss],axis=0)
                c += 1
        
        FIMs = tf.squeeze(FIMs)
        Losses = tf.squeeze(Losses)
        
        #K is equal to the lowest curvature at the loss of 1/number of classes
        change_loss_squared = (Losses - 1/self.num_classes)**2 #get the squared difference from the loss of 1/num_classes
        lowest_change_loss, lowest_change_loss_idx = tf.math.approx_min_k(change_loss_squared,self.filter_strength) #get the index of the lowest change in loss
        K_lowest_loss = tf.gather(FIMs,lowest_change_loss_idx) #get the loss at the lowest change in loss
        self.K = tf.reduce_mean(K_lowest_loss) #get the average loss at the lowest change in loss
        self.K_conf = tf.reduce_mean(lowest_change_loss) #get the average confidence at the lowest change in loss
        print("K: ",K)
        print("K_conf: ",K_conf)

        #save to file
        if self.save:
            self.df = pd.concat([self.df,pd.DataFrame({"FIM":FIMs.numpy(),"Loss":Losses.numpy()})],ignore_index=True,axis=1)
            self.df.to_csv(str(self.prefix)+"K"+self.K+".csv")
    
    def _K_Predict(self, data):
        imgs = data[0] #this is a batch of images
        labels = data[1] #this is a batch of labels
        probs = self.model(imgs,training=False)
        #append the bool values 
        self.OG_acc_accum = tf.concat([self.OG_acc_accum,tf.argmax(probs,axis=1) == tf.argmax(labels,axis=1)],axis=0)

        #calc FIM for each ouput of each image
        c = 0
        for class_idx in range(self.num_classes):
            batched_class_idx = tf.ones((self.get_bs(imgs),1))*class_idx
            batched_class_idx = tf.cast(batched_class_idx,tf.int32)
            F = self.Get_Z_sm_class(imgs,batched_class_idx) #FIM for batch of images for specific class
            F = tf.expand_dims(F,1)
            if c == 0:
                FIMs = F
                c += 1
            else:
                FIMs = tf.concat([FIMs,F],axis=1)
                c += 1
        #FIMs should be [num_images*num_classes]
        FIMs = FIMs.numpy()

        #get the lowest curvature for each image
        lowest_curvature = np.min(FIMs,axis=1)
        #get the index of the lowest curvature
        lowest_curvature_idx = np.argmin(FIMs,axis=1)
        #get the accuracy of the model using the lowest curvature
        self.lowest_C_acc_accum = tf.concat([self.lowest_C_acc_accum,lowest_curvature_idx == tf.argmax(labels,axis=1)],axis=0)
    
    def _K_Predict_End(self):
        OG_acc_accum = tf.cast(self.OG_acc_accum,tf.float32)
        lowest_C_acc_accum = tf.cast(self.lowest_C_acc_accum,tf.float32)
        # if both 1 then = 1 else 0
        combined = tf.cast(tf.math.logical_and(self.OG_acc_accum,self.lowest_C_acc_accum),tf.float32)

        OG_acc = tf.reduce_mean(OG_acc_accum)
        lowest_C_acc = tf.reduce_mean(lowest_C_acc_accum)
        combined_acc = tf.reduce_mean(combined)
        print("Original Accuracy: ",OG_acc)
        print("Lowest Curvature Accuracy: ",lowest_C_acc)
        print("Combined Accuracy: ",combined_acc)
        
class LogOutsFIM(keras.callbacks.Callback):
    def __init__(self, ds, epochRecord,loss_func, limit=500,FIM_type='stat',prefix=""):
        self.ds = ds
        self.epoch = 0
        self.epochRecord = epochRecord
        self.limit = limit
        self.df = pd.DataFrame(columns=['id'])
        self.loss_function = loss_func
        self.prefix = prefix
        self.FIM_type = FIM_type

    @tf.function
    def Get_Z_sm_flat(self,items):
        x,y = items
        bs = tf.shape(x)[0]
        
        with tf.GradientTape() as tape:
            #print(text_batch[i])
            y_hat = self.model(x,training=False) #get model output (softmax) [BS x num_classes]
            loss = self.loss_function(y,y_hat)
            #selected = tf.squeeze(tf.random.categorical(tf.math.log(y_hat), 1)) #sample from the output [BS x 1]
            #select based on flat distribution
            selected = tf.squeeze(tf.random.uniform((bs,),0,10,dtype=tf.int64))
            output = tf.gather(y_hat,selected,axis=1,batch_dims=1) #get the output for the selected class [BS x 1]
            output = tf.math.log(output)#tf.math.log(output) #log the output [BS x 1]

        j = tape.jacobian(output,self.model.trainable_variables)
        layer_sizes = [tf.reduce_sum(tf.size(v)) for v in self.model.trainable_variables] #get the size of each layer
        j = [tf.reshape(j[i],(bs,layer_sizes[i])) for i in range(len(j))] #reshape the gradient to [BS x num_layer_params x layers]
        j = tf.concat(j,axis=1) #concat the gradient over the layers [BS x num_params]
        j = tf.square(j) #square the gradient [BS x num_params]
        j = tf.reduce_sum(j,axis=1) #sum the gradient [ 1]
        return j, output
    
    @tf.function
    def Get_Z_sm_stat(self,items):
        x,y = items
        bs = tf.shape(x)[0]
        
        with tf.GradientTape() as tape:
            #print(text_batch[i])
            y_hat = self.model(x,training=False) #get model output (softmax) [BS x num_classes]
            loss = self.loss_function(y,y_hat)
            selected = tf.squeeze(tf.random.categorical(tf.math.log(y_hat), 1)) #sample from the output [BS x 1]
            #select based on flat distribution
            #selected = tf.squeeze(tf.random.uniform((bs,),0,10,dtype=tf.int64))
            output = tf.gather(y_hat,selected,axis=1,batch_dims=1) #get the output for the selected class [BS x 1]
            output = tf.math.log(output)#tf.math.log(output) #log the output [BS x 1]

        j = tape.jacobian(output,self.model.trainable_variables)
        layer_sizes = [tf.reduce_sum(tf.size(v)) for v in self.model.trainable_variables] #get the size of each layer
        j = [tf.reshape(j[i],(bs,layer_sizes[i])) for i in range(len(j))] #reshape the gradient to [BS x num_layer_params x layers]
        j = tf.concat(j,axis=1) #concat the gradient over the layers [BS x num_params]
        j = tf.square(j) #square the gradient [BS x num_params]
        j = tf.reduce_sum(j,axis=1) #sum the gradient [ 1]
        return j, output
    
    @tf.function
    def Get_Z_sm_emp(self,items):
        x,y = items
        bs = tf.shape(x)[0]
        #print(y)
        
        with tf.GradientTape() as tape:
            #print(text_batch[i])
            y_hat = self.model(x,training=False) #get model output (softmax) [BS x num_classes]
            loss = self.loss_function(y,y_hat)
            #selected = tf.squeeze(tf.random.categorical(tf.math.log(y_hat), 1)) #sample from the output [BS x 1]
            #select based on flat distribution
            #selected = tf.squeeze(tf.random.uniform((bs,),0,10,dtype=tf.int64))
            output = tf.gather(y_hat,tf.argmax(y,axis=1),axis=1,batch_dims=1) #get the output for the selected class [BS x 1]
            output = tf.math.log(output)#tf.math.log(output) #log the output [BS x 1]

        j = tape.jacobian(output,self.model.trainable_variables)
        layer_sizes = [tf.reduce_sum(tf.size(v)) for v in self.model.trainable_variables] #get the size of each layer
        j = [tf.reshape(j[i],(bs,layer_sizes[i])) for i in range(len(j))] #reshape the gradient to [BS x num_layer_params x layers]
        j = tf.concat(j,axis=1) #concat the gradient over the layers [BS x num_params]
        j = tf.square(j) #square the gradient [BS x num_params]
        j = tf.reduce_sum(j,axis=1) #sum the gradient [ 1]
        return j, output
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch in self.epochRecord):
            print("doing LossFIMRecord")
            c = 0
            for items in self.ds:
                if c*items[0].shape[0] > self.limit:
                    break
                if self.FIM_type == 'flat':
                    j, out = self.Get_Z_sm_flat(items)
                elif self.FIM_type == 'stat':
                    j, out = self.Get_Z_sm_stat(items)
                elif self.FIM_type == 'emp':
                    j, out = self.Get_Z_sm_emp(items)
                else:
                    assert False, "FIM type not recognized"
                if c == 0:
                    FIMs = j
                    Outs = out
                    c += 1
                else:
                    FIMs = tf.concat([FIMs,j],axis=0)
                    Outs = tf.concat([Outs,out],axis=0)
                    c += 1
            
            FIMs = tf.squeeze(FIMs)
            Outs = tf.squeeze(Outs)
            #add the FIMs to the dataframe
            self.df = pd.concat([self.df,pd.DataFrame({str(epoch)+"FIM":FIMs.numpy(),str(epoch)+"logOutput":Outs.numpy()})],ignore_index=True,axis=1)
            
            #self.df = self.df.append({str(self.epoch)+"FIM":FIMs[i].numpy(),str(self.epoch)+"Loss":Losses[i].numpy()},ignore_index=True)
    
    def on_train_end(self, logs=None):
        self.df.to_csv(str(self.prefix)+"LogOutFIM.csv")

class CustomEarlyStopping(keras.callbacks.Callback):
    def __init__(self):
        #early stop if the categorical accuracy is equals 1
        self.a = 1

        
    def on_epoch_end(self, epoch, logs=None):
        if logs["categorical_accuracy"] == 1:
            self.model.stop_training = True
        


def main(epochs, n, bs,opt,lr):
    #dowload dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    #normalize the data
    x_train = x_train/255
    x_test = x_test/255

    #add another dimention containing if its mislabeled
    #y_train = np.concatenate((y_train, np.zeros((y_train.shape[0],1))), axis=1)

    #randomly mislabel n% of the data
    # for i in range(y_train.shape[0]):
    #     if np.random.rand() < n:
    #         y_train[i,1] = 1 #1 means mislabeled
    #         y_train[i,0] = np.random.randint(0,10)

    #randomly blur n% of the data
    # for i in range(y_train.shape[0]):
    #     if np.random.rand() < n:
    #         y_train[i,1] = 1
    #         x_train[i] = tf.random.normal(x_train[i].shape, mean=0.5, stddev=0.5, dtype=tf.float64)
    print(x_train.shape)

    #x_train_blur_025 = [x + tf.random.normal(x.shape, mean=0, stddev=0.25, dtype=tf.float64) for x in x_train[:2000]]
    #x_train_blur_075 = [x + tf.random.normal(x.shape, mean=0, stddev=0.75, dtype=tf.float64) for x in x_train[:2000]]

    #ensure the data is between 0 and 1
    #x_train_blur_025 = [tf.clip_by_value(x, 0, 1) for x in x_train_blur_025]
    #x_train_blur_075 = [tf.clip_by_value(x, 0, 1) for x in x_train_blur_075]
    #print(x_train_blur

    #make dataset
    #correct_train = tf.data.Dataset.from_tensor_slices((x_train[y_train[:,1] == 0], y_train[y_train[:,1] == 0][:,0]))
    #mislabeled_train = tf.data.Dataset.from_tensor_slices((x_train[y_train[:,1] == 1], y_train[y_train[:,1] == 1][:,0]))
    #random_dataset = tf.data.Dataset.from_tensor_slices((np.random.rand(5000, 32, 32, 3), np.random.choice(10, 5000)))
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)) #[:,0]
    #train_subset = tf.data.Dataset.from_tensor_slices((x_train[:2], y_train[:2]))
    #train_blur_025 = tf.data.Dataset.from_tensor_slices((x_train_blur_025, y_train[:2000]))
    #train_blur_075 = tf.data.Dataset.from_tensor_slices((x_train_blur_075, y_train[:2000]))
    
    
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    #Filtering
    #train_dataset = train_dataset.filter(lambda img, label: tf.squeeze((label == 0) or (label == 1)))
    #test_dataset = test_dataset.filter(lambda img, label: tf.squeeze((label == 0) or (label == 1)))

    def map_fn(image, label):
        #image = tf.cast(image, tf.float32)
        #image = tf.expand_dims(image, -1)
        return image, tf.squeeze(tf.one_hot(tf.cast(label,tf.int32), 10,on_value=1.0))

    train_dataset = train_dataset.map(map_fn)
    #train_subset = train_subset.map(map_fn)
    #correct_train = correct_train.map(map_fn)
    #mislabeled_train = mislabeled_train.map(map_fn)
    test_dataset = test_dataset.map(map_fn)
    #random_dataset = random_dataset.map(map_fn)
    #train_blur_025 = train_blur_025.map(map_fn)
    #train_blur_075 = train_blur_075.map(map_fn)


    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(bs)
    #correct_train = correct_train.batch(bs)
    #mislabeled_train = mislabeled_train.batch(bs)
    test_dataset = test_dataset.batch(bs)
    #train_subset = train_subset.batch(bs)
    #random_dataset = random_dataset.batch(bs)
    #train_blur_025 = train_blur_025.batch(bs)
    #train_blur_075 = train_blur_075.batch(bs)

    # Get an item from the test_dataset
    #print(next(iter(test_dataset))[1].shape)
    #print(next(iter(train_blur))[0].shape)
    #print(next(iter(correct_train))[1].shape)
    #print(next(iter(mislabeled_train))[1].shape)
    train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


    #create model
    model = tf.keras.Sequential([
        #tf.keras.layers.Flatten(input_shape=(32,32,3)),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'), #64
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'), #64
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'), #64
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    # model = tf.keras.Sequential([
    #     #tf.keras.layers.Flatten(input_shape=(32,32,3)),
    #     tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    #     tf.keras.layers.Conv2D(8, (3,3), activation='relu'), #64
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(32, activation='relu'), #64
    #     tf.keras.layers.Dense(10, activation='softmax')
    # ])


    #lr_schedule = lr_schedules.StepChange([0,5,30],[0.005,0.001,0.01],100,1563)
    #lr_schedule = lr_schedules.LinearChange(0.01,0.0001,100,1563)
    # lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    #         0.01,
    #         1563*10,
    #         t_mul=2.0,
    #         m_mul=1.0,
    #         alpha=0.0,
    #         name='SGDRDecay'
    #     )
    #lr_schedule = 0.001
    #lr = 0.1

    if opt == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif opt == "SGD":
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.2)
    else:
        print("Invalid optimizer")
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    nored_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy()

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[train_acc_metric])

    #FIMMisslabel = GFIMCallback(mislabeled_train, False, 4, nored_loss_fn, False, "Misslabel")
    #FIMCorrect = GFIMCallback(correct_train, False, 4, nored_loss_fn, False, "Correct")
    #FIMAll = GFIMCallback(train_dataset, False, 8, nored_loss_fn, False, "All")
    #LossFIM = LossFIMCallback(train_dataset, [0,20,40,60,80,99], nored_loss_fn, limit=5000,prefix="CCEoff-0_0001",do_Mag=False)
    #corrCallback = CorrectLossFIMCallback(train_dataset, np.arange(100), nored_loss_fn, limit=5000,prefix="alphaCalcs")
    #AugCallback = AugmentLossFIMCallback(train_subset, [0,20,40,60,80,99], nored_loss_fn, limit=5000,prefix="AugmentTradNoise")
    #SubsetLossFIM = LossFIMCallback(train_subset, [80], nored_loss_fn, limit=10000,prefix="NormalBatch",do_batch=True)
    #corrLossFIM = LossFIMCallback(correct_train, [0,20,40,60,80,100], nored_loss_fn, limit=5000,prefix="RandomCorrect")
    #gradMag = gradMagCallback(train_dataset, [0,20,40,60,80,99], nored_loss_fn, limit=5000,prefix="Normal")
    WandbCallback = wandb.keras.WandbCallback(save_model=False)
    #CalcK = Calc_K_On_End(train_dataset, nored_loss_fn, limit=5000,save=False,prefix="Test")
    #FIMMaxClass = LossFIMMaxClassOutput([i for i in range(0,101,5)], nored_loss_fn, limit=5000,prefix="MaxClassOutput",classes=10)
    #LogOutsStat = LogOutsFIM(train_dataset, [i for i in range(0,101,5)], nored_loss_fn, limit=2000,FIM_type='stat',prefix="typeStat")
    LogOutsFlat = LogOutsFIM(train_dataset, [i for i in range(0,101,5)], nored_loss_fn, limit=2000,FIM_type='flat',prefix="OptADAMLR0_0001")
    #LogOutsEmp = LogOutsFIM(train_dataset, [i for i in range(0,101,5)], nored_loss_fn, limit=2000,FIM_type='emp',prefix="typeEmp")
    #LogOutstest = LogOutsFIM(test_dataset, [i for i in range(0,101,5)], nored_loss_fn, limit=5000,prefix="testFIMFlatDistShortModel")
    #LogOutsrand = LogOutsFIM(random_dataset, [i for i in range(0,101,5)], nored_loss_fn, limit=4999,prefix="RandFIMSampDist")
    #LogOutsrBlur025 = LogOutsFIM(train_blur_025, [i for i in range(0,101,5)], nored_loss_fn, limit=2000,prefix="Blur025FIMFlatDist")
    #LogOutsrBlur075 = LogOutsFIM(train_blur_075, [i for i in range(0,101,5)], nored_loss_fn, limit=2000,prefix="Blur075FIMFlatDist")
    #LayerLossFIM = LayerLossFIMCallback(train_dataset, np.arange(100), nored_loss_fn, limit=2000,prefix="LayerAlpha")
    #TestLayerLossFIM = LayerLossFIMCallback(test_dataset, np.arange(100), nored_loss_fn, limit=2000,prefix="TestLayerAlpha")

    #CES = CustomEarlyStopping()

    #train the model
    model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, callbacks=[LogOutsFlat,WandbCallback])

    #evaluate the model on the test set with normal predictions and K predictions
    # c=0
    # for test_batch in test_dataset:
    #     c+=1
    #     print(c)
    #     CalcK._K_Predict(test_batch)
    #     if c > 10:
    #         break
    # CalcK._K_Predict_End()




def csv_to_graphs():
    df = pd.read_csv("MNISTNormalLossFIM.csv")
    #df2 = pd.read_csv("RandomCorrectLossFIM.csv")
    fim_idx = [1,3,5,7,9]
    loss_idx = [2,4,6,8,10]
    epochs = [0,20,40,60,80]
    for i in range(len(fim_idx)):
        plt.scatter(df[str(fim_idx[i])],df[str(loss_idx[i])],s=1,alpha=0.2,color="red")
        #plt.scatter(df2[str(fim_idx[i])],df2[str(loss_idx[i])],s=1,alpha=0.2,color="green")
        plt.xlabel("FIM")
        plt.ylabel("Loss")
        plt.title("FIM vs Loss at epoch "+str(epochs[i]))
        #log x axis
        plt.xscale("log")
        plt.yscale("log")
        #axis limits
        plt.xlim(10e-10,10e5) #(10e-10,10e5)
        plt.ylim(10e-8,10e1) #(10e-8,10e1)
        
        #add grid lines
        plt.grid()
        plt.savefig("MNISTFIMvsLoss_"+str(epochs[i])+".png")
        #clear plot
        plt.clf()

if __name__ == "__main__":
    #csv_to_graphs()
    #prnt("done")
    os.environ['WANDB_API_KEY'] = 'fc2ea89618ca0e1b85a71faee35950a78dd59744'
    wandb.login()
    wandb.init(project="MisslabelFIM",name="SGDLR0_001mom0_2")
    wandb.config.epochs = 100
    wandb.config.n = 0.2
    wandb.config.bs = 32
    wandb.config.lr = 0.0001
    wandb.config.opt = "Adam"
    main(wandb.config.epochs, wandb.config.n, wandb.config.bs, wandb.config.opt, wandb.config.lr)
