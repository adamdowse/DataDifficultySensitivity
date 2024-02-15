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
    wandb.init(project='GFIMUseage',config=config.__dict__)
    #dataset = DataHandler.DataHandler(config)


    #load data
    train_ds = tfds.load(name="mnist", split="train") #This should be a tf.data.Dataset
    test_ds = tfds.load(name="mnist", split="test")

    #preprocess data
    def preprocess(item):
        img = item['image']
        img = tf.cast(img,tf.float32)/255.0
        label = item['label']
        return img,label

    train_ds = train_ds.map(preprocess)
    test_ds = test_ds.map(preprocess)

    #create model
    model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(64,3,activation='relu'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128,activation='relu'),
            tf.keras.layers.Dense(self.num_classes,activation='softmax')
        ])

    #compile model with optimizer and loss function
    model.compile(optimizer='SGD',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    GROUPS = 10

    def record_losses(ds,ds_datacount,bs,model):
        #record losses and add to ds
        iter_ds = iter(ds)
        losses = np.zeros(ds_datacount)
        #do in batches
        for i in range(ds_size):
            x,y = next(iter_ds)
            loss = model.evaluate(x,y)
            losses[i*bs:(i+1)*bs] = loss
        
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
            wandb.log({'loss_group'+str(i):grouped_losses},step=epoch)


        #add losses to ds
        loss_ds = tf.data.Dataset.from_tensor_slices(losses)
        ds = tf.data.Dataset.zip((ds,loss_ds)) #this should be a tuple of (x,y,loss)
        return ds, (min_losses,max_losses,group_size) #TODO chenage this to handle different gourp sizes

    def split_GFIM_ds(ds,model,loss_info,group_num): #ds is a tuple of (x,y,loss)
        #record GFIM
        sub_ds = ds.filter(lambda x,y,loss: if loss >= loss_info[0][group_num] and loss <= loss_info[1][group_num])
        return sub_ds

    @tf.function
    def Get_Z_single(item):
        img,label,loss = item
        with tf.GradientTape() as tape:
            y_hat = model(img,training=False) #[0.1,0.8,0.1,ect] this is output of softmax
        selected = tf.squeeze(tf.random.categorical(tf.math.log(y_hat), 1)) #[2]
        #output = tf.gather(y_hat,selected,axis=1,batch_dims=1)
        output = tf.gather(y_hat,selected,axis=1) #[0.3]
        output = tf.math.log(output)
        g = tape.jacobian(output,.model.trainable_variables)
        layer_sizes = [tf.reduce_sum(tf.size(v)) for v in model.trainable_variables]
        g = [tf.reshape(g[i],(layer_sizes[i])) for i in range(len(g))] #TODO check that this dosent need to deal with batches
        g = tf.concat(g,axis=1)
        g = tf.square(g)
        g = tf.reduce_sum(g)
        return g

    def record_GFIM(ds,model,group_num):
        iter_ds = iter(ds)
        data_count = 0
        mean = 0
        iter_ds = iter(ds)
        for _ in range(lower_lim):
            data_count += 1
            if data_count % 500 == 0:
                print(data_count)
            x = Get_Z_single(next(iter_ds)) #just one replica can be used here
            delta = x - mean 
            mean += delta/(data_count)
        print('--> time: ',time.time()-t)
        wandb.log({'GFIM_'+str(group_num):mean},step=epoch)
        return mean

    def reduce_ds(ds,GFIM_history,loss_info):
        #are any groups reducing in GFIM
        removed_groups = []
        for i in range(GROUPS):
            if GFIM_history[-1][i] < GFIM_history[-2][i]:
                removed_groups.append(i)
                ds = ds.filter(lambda x,y,loss: if loss >= loss_info[0][i] and loss <= loss_info[1][i])
                wandb.log({'group_used'+str(i):0},step=epoch)
            else:
                wandb.log({'group_used'+str(i):1},step=epoch)
        if len(removed_groups) == 0:
            print('No groups removed')
        elif len(removed_groups) == GROUPS:
            print('All groups removed')
        return ds


        
        

    GFIM_history = np.zeros(1,GROUPS)#[[0,0,0,0,0,0,0,0,0,0]] -> [[0,0,0,0,0,0,0,0,0,0],[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]] ect

    max_epoch = 20
    burnin_epochs = 2
    for epoch in range(max_epochs):
        if epoch < burnin_epochs:
            loss_ds, loss_info = record_losses(train_ds,60000,32,model)
            temp_GFIM_history = np.zeros(GROUPS)
            for i in range(GROUPS):
                sub_ds = split_GFIM_ds(loss_ds,model,loss_info,i)
                g = record_GFIM(sub_ds,model)
                temp_GFIM_history[i] = g
            GFIM_history = np.vstack((GFIM_history,temp_GFIM_history))
            #train on full dataset
            hist = model.fit(train_ds,epochs=1)
            wandb.log(hist.history,step=epoch)

        else:
            loss_ds, loss_info = record_losses(train_ds,60000,32,model)
            for i in range(GROUPS):
                sub_ds = split_GFIM_ds(loss_ds,model,loss_info,i)
                record_GFIM(sub_ds,model)
            red_ds = reduce_ds(train_ds,GFIM_history,loss_info)
            #train on reduced dataset
            hist = model.fit(red_ds,epochs=1)
            wandb.log(hist.history,step=epoch)
        
        hist = model.evaluate(test_ds)
        wandb.log(hist.history,step=epoch)


if __name__ == "__main__":
    os.environ['WANDB_API_KEY'] = 'fc2ea89618ca0e1b85a71faee35950a78dd59744'
    wandb.login()
    Main()