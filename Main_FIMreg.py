#This is the main file and should be used to run the project

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import os
import Models
import DataHandlerV2 as DataHandler
import wandb
from tensorflow import keras
from wandb.keras import WandbCallback
import time
import tracemalloc
import os
import argparse



def Main():
    print ("Main Started")
    
    #setup
    tf.keras.backend.clear_session()
    wandb.init(project='GFIMUseage')
    #dataset = DataHandler.DataHandler(config)

    GROUPS = 10
    BS = 32
    data_count = 50000

    #load data
    train_ds = tfds.load(name="mnist", split="train") #This should be a tf.data.Dataset
    test_ds = tfds.load(name="mnist", split="test")

    #preprocess data
    def preprocess(item):
        img = item['image']
        img = tf.cast(img,tf.float32)/255.0
        #convert label to onehot
        label = tf.one_hot(item['label'],10)
        return img,label

    train_ds = train_ds.map(preprocess)
    test_ds = test_ds.map(preprocess)

    train_ds = train_ds.batch(BS)
    test_ds = test_ds.batch(BS)

    #create model
    initializer = tf.keras.initializers.GlorotNormal(seed=42)
    model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=(28,28,1)),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(64,3,activation='relu'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128,activation='relu'),
            tf.keras.layers.Dense(10,activation='softmax')
        ])

    #compile model with optimizer and loss function
    model.compile(optimizer='SGD',loss='categorical_crossentropy',metrics=['accuracy'])
    #setinitial learning rate
    model.optimizer.lr = 0.001
    model.summary()

    def record_losses(ds,ds_datacount,bs,model):
        print('recording losses')
        #ds =ds.unbatch()
        #ds = ds.batch(1)
        #record losses and add to ds
        iter_ds = iter(ds)
        losses = np.zeros(ds_datacount)
        for i in range(ds_datacount//bs):
            x,y = next(iter_ds)
            with tf.GradientTape() as tape:
                y_hat = model(x,training=False)
                loss = tf.keras.losses.categorical_crossentropy(y,y_hat)
            losses[i*bs:i*bs + len(loss)] = loss

        #group losses into n groups
        oredered_losses = np.sort(losses)
        group_size = int(ds_datacount/GROUPS)
        grouped_losses = np.zeros(GROUPS)
        min_losses = np.zeros(GROUPS)
        max_losses = np.zeros(GROUPS)
        for i in range(GROUPS):
            min_losses[i] = oredered_losses[i*group_size]
            max_losses[i] = oredered_losses[(i+1)*group_size-1]
            grouped_losses[i] = np.mean(oredered_losses[i*group_size:(i+1)*group_size])
            wandb.log({'loss_groups'+str(i):sum(grouped_losses)},step=epoch)

        #add losses to ds
        ds = ds.unbatch()
        loss_ds = tf.data.Dataset.from_tensor_slices(losses)
        ds = tf.data.Dataset.zip(ds,loss_ds).map(lambda og,loss: (og[0],og[1],loss))
        return ds, (min_losses,max_losses,group_size)

    def split_GFIM_ds(ds,loss_info,group_num): #ds is a tuple of (x,y,loss)
        print('splitting GFIM ds')
        #record GFIM
        sub_ds = ds.filter(lambda x,y,loss: loss >= loss_info[0][group_num] and loss <= loss_info[1][group_num])
        return sub_ds

    #@tf.function
    def Get_Z_single(item):
        img,label,loss = item
        with tf.GradientTape() as tape:
            y_hat = model(img,training=False) #[0.1,0.8,0.1,ect] this is output of softmax
            output = tf.squeeze(tf.random.categorical(tf.math.log(y_hat), 1)) #[2]
            #output = tf.gather(y_hat,selected,axis=1,batch_dims=1)
            output = tf.gather(y_hat,output,axis=1) #[0.3]
            output = tf.squeeze(output)
            output = tf.math.log(output)
        g = tape.gradient(output,model.trainable_variables)#This or Jacobian?
        
        layer_sizes = [tf.reduce_sum(tf.size(v)) for v in model.trainable_variables]
        g = [tf.reshape(g[i],(layer_sizes[i])) for i in range(len(g))] #TODO check that this dosent need to deal with batches
        g = tf.concat(g,axis=0)
        g = tf.square(g)
        g = tf.reduce_sum(g)
        return g

    def record_GFIM(ds,model,group_num,group_info):
        print('recording GFIM')
        ds = ds.batch(1)
        data_count = 0
        mean = 0
        iter_ds = iter(ds)
        low_lim = min(group_info[2],1000)
        for _ in range(low_lim):
            data_count += 1
            if data_count % 500 == 0:
                print(data_count)
            x = Get_Z_single(next(iter_ds)) #just one replica can be used here
            delta = x - mean 
            mean += delta/(data_count)
        wandb.log({'GFIM_'+str(group_num):mean},step=epoch)
        return mean

    def reduce_ds(ds,GFIM_history,loss_info):
        #are any groups reducing in GFIM
        removed_groups = []
        for i in range(GROUPS):
            if GFIM_history[-1][i] < GFIM_history[-2][i]: #If reducing
                removed_groups.append(i)
                ds = ds.filter(lambda x,y,loss: loss < loss_info[0][i] or loss > loss_info[1][i]) #selcet all but group i
                wandb.log({'group_used'+str(i):0},step=epoch)
            else:
                wandb.log({'group_used'+str(i):1},step=epoch)
        
        print('Groups removed:',removed_groups)
        iter_ds = iter(ds)
        x,y,loss = next(iter_ds)
        print(x.shape)
        print(y.shape)
        print(loss.shape)
        ds = ds.map(lambda x,y,loss: (x,y)).batch(32)
        return ds

    GFIM_history = np.zeros((1,GROUPS))#[[0,0,0,0,0,0,0,0,0,0]] -> [[0,0,0,0,0,0,0,0,0,0],[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]] ect

    max_epochs = 20
    burnin_epochs = 2

    for epoch in range(max_epochs):
        print('Epoch:',epoch)
        if epoch < burnin_epochs:
            loss_ds, loss_info = record_losses(train_ds,data_count,BS,model)
            temp_GFIM_history = np.zeros(GROUPS)
            for i in range(GROUPS):
                sub_ds = split_GFIM_ds(loss_ds,loss_info,i)
                g = record_GFIM(sub_ds,model,i,loss_info)
                temp_GFIM_history[i] = g
            GFIM_history = np.vstack((GFIM_history,temp_GFIM_history))
            #train on full dataset
            hist = model.fit(train_ds,epochs=1)
            wandb.log({'loss':hist.history['loss'],'accuracy':hist.history['accuracy']},step=epoch)

        else:
            loss_ds, loss_info = record_losses(train_ds,data_count,32,model)
            temp_GFIM_history = np.zeros(GROUPS)
            for i in range(GROUPS):
                sub_ds = split_GFIM_ds(loss_ds,loss_info,i)
                g = record_GFIM(sub_ds,model,i,loss_info)
                temp_GFIM_history[i] = g
            GFIM_history = np.vstack((GFIM_history,temp_GFIM_history))
            red_ds = reduce_ds(loss_ds,GFIM_history,loss_info)
            #train on reduced dataset
            
            hist = model.fit(red_ds,epochs=1)
            wandb.log({'loss':hist.history['loss'],'accuracy':hist.history['accuracy']},step=epoch)
        
        hist = model.evaluate(test_ds)
        wandb.log({'test_loss':hist[0],'test_accuracy':hist[1]},step=epoch)


def Standard_Main():
    print ("Main Started")
    
    #setup
    tf.keras.backend.clear_session()
    wandb.init(project='GFIMUseage')
    #dataset = DataHandler.DataHandler(config)

    GROUPS = 10
    BS = 32
    data_count = 50000

    #load data
    train_ds = tfds.load(name="mnist", split="train") #This should be a tf.data.Dataset
    test_ds = tfds.load(name="mnist", split="test")

    #preprocess data
    def preprocess(item):
        img = item['image']
        img = tf.cast(img,tf.float32)/255.0
        #convert label to onehot
        label = tf.one_hot(item['label'],10)
        return img,label

    train_ds = train_ds.map(preprocess)
    test_ds = test_ds.map(preprocess)

    train_ds = train_ds.batch(BS)
    test_ds = test_ds.batch(BS)

    #create model
    initializer = tf.keras.initializers.GlorotNormal(seed=42)
    model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=(28,28,1)),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(64,3,activation='relu'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128,activation='relu'),
            tf.keras.layers.Dense(10,activation='softmax')
        ])

    #compile model with optimizer and loss function
    model.compile(optimizer='SGD',loss='categorical_crossentropy',metrics=['accuracy'])
    #setinitial learning rate
    model.optimizer.lr = 0.001
    model.summary()
    
    
    def record_losses(ds,ds_datacount,bs,model):
        print('recording losses')
        #ds =ds.unbatch()
        #ds = ds.batch(1)
        #record losses and add to ds
        iter_ds = iter(ds)
        losses = np.zeros(ds_datacount)
        for i in range(ds_datacount//bs):
            x,y = next(iter_ds)
            with tf.GradientTape() as tape:
                y_hat = model(x,training=False)
                loss = tf.keras.losses.categorical_crossentropy(y,y_hat)
            losses[i*bs:i*bs + len(loss)] = loss

        #group losses into n groups
        oredered_losses = np.sort(losses)
        group_size = int(ds_datacount/GROUPS)
        grouped_losses = np.zeros(GROUPS)
        min_losses = np.zeros(GROUPS)
        max_losses = np.zeros(GROUPS)
        for i in range(GROUPS):
            min_losses[i] = oredered_losses[i*group_size]
            max_losses[i] = oredered_losses[(i+1)*group_size-1]
            grouped_losses[i] = np.mean(oredered_losses[i*group_size:(i+1)*group_size])
            wandb.log({'loss_groups'+str(i):sum(grouped_losses)},step=epoch)

        #add losses to ds
        ds = ds.unbatch()
        loss_ds = tf.data.Dataset.from_tensor_slices(losses)
        ds = tf.data.Dataset.zip(ds,loss_ds).map(lambda og,loss: (og[0],og[1],loss))
        return ds, (min_losses,max_losses,group_size)

    def split_GFIM_ds(ds,loss_info,group_num): #ds is a tuple of (x,y,loss)
        print('splitting GFIM ds')
        #record GFIM
        sub_ds = ds.filter(lambda x,y,loss: loss >= loss_info[0][group_num] and loss <= loss_info[1][group_num])
        return sub_ds

    #@tf.function
    def Get_Z_single(item):
        img,label,loss = item
        with tf.GradientTape() as tape:
            y_hat = model(img,training=False) #[0.1,0.8,0.1,ect] this is output of softmax
            output = tf.squeeze(tf.random.categorical(tf.math.log(y_hat), 1)) #[2]
            #output = tf.gather(y_hat,selected,axis=1,batch_dims=1)
            output = tf.gather(y_hat,output,axis=1) #[0.3]
            output = tf.squeeze(output)
            output = tf.math.log(output)
        g = tape.gradient(output,model.trainable_variables)#This or Jacobian?
        
        layer_sizes = [tf.reduce_sum(tf.size(v)) for v in model.trainable_variables]
        g = [tf.reshape(g[i],(layer_sizes[i])) for i in range(len(g))] #TODO check that this dosent need to deal with batches
        g = tf.concat(g,axis=0)
        g = tf.square(g)
        g = tf.reduce_sum(g)
        return g

    def record_GFIM(ds,model,group_num,group_info):
        print('recording GFIM')
        ds = ds.batch(1)
        data_count = 0
        mean = 0
        iter_ds = iter(ds)
        low_lim = min(group_info[2],1000)
        for _ in range(low_lim):
            data_count += 1
            if data_count % 500 == 0:
                print(data_count)
            x = Get_Z_single(next(iter_ds)) #just one replica can be used here
            delta = x - mean 
            mean += delta/(data_count)
        wandb.log({'GFIM_'+str(group_num):mean},step=epoch)
        return mean

    def reduce_ds(ds,GFIM_history,loss_info):
        #are any groups reducing in GFIM
        removed_groups = []
        for i in range(GROUPS):
            if GFIM_history[-1][i] < GFIM_history[-2][i]: #If reducing
                removed_groups.append(i)
                ds = ds.filter(lambda x,y,loss: loss < loss_info[0][i] or loss > loss_info[1][i]) #selcet all but group i
                wandb.log({'group_used'+str(i):0},step=epoch)
            else:
                wandb.log({'group_used'+str(i):1},step=epoch)
        
        print('Groups removed:',removed_groups)
        iter_ds = iter(ds)
        x,y,loss = next(iter_ds)
        print(x.shape)
        print(y.shape)
        print(loss.shape)
        ds = ds.map(lambda x,y,loss: (x,y)).batch(32)
        return ds

    GFIM_history = np.zeros((1,GROUPS))#[[0,0,0,0,0,0,0,0,0,0]] -> [[0,0,0,0,0,0,0,0,0,0],[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]] ect

    max_epochs = 20
    burnin_epochs = 2

    for epoch in range(max_epochs):
        print('Epoch:',epoch)
        loss_ds, loss_info = record_losses(train_ds,data_count,BS,model)
        for i in range(GROUPS):
            sub_ds = split_GFIM_ds(loss_ds,loss_info,i)
            g = record_GFIM(sub_ds,model,i,loss_info)
        #train on full dataset
        hist = model.fit(train_ds,epochs=1)
        wandb.log({'loss':hist.history['loss'],'accuracy':hist.history['accuracy']},step=epoch)
        hist = model.evaluate(test_ds)
        wandb.log({'test_loss':hist[0],'test_accuracy':hist[1]},step=epoch)


if __name__ == "__main__":
    os.environ['WANDB_API_KEY'] = 'fc2ea89618ca0e1b85a71faee35950a78dd59744'
    wandb.login()
    #Main()
    Standard_Main()