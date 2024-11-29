import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras

import wandb 
import os


#redcord the GFIM for a training run of each medical dataset.
#- Heart desease
#- HAM10000
#- Brain Tumour segmentation



class GFIMCallback(keras.callbacks.Callback):
    def __init__(self, ds, doStep,num_groups,loss_func):
        self.doStep = doStep
        self.ds = ds
        self.num_groups = num_groups
        self.epoch = 0
        self.curr_batch = 0
        self.max_batch = 0
        self.loss_function = loss_func

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
        j = j/tf.cast(tf.reduce_sum(layer_sizes),tf.float32) #normalize the gradient [ 1]
        return j, loss

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0:
            self.max_batch = self.curr_batch
        self.epoch = epoch
        if not self.doStep:
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
                
                wandb.log({"GFIM_"+str(i):tf.reduce_mean(tf.gather(FIMs,group_idx),axis=0)},step=epoch)
                wandb.log({"Loss_"+str(i):tf.reduce_mean(tf.gather(Losses,group_idx),axis=0)},step=epoch)
            wandb.log({"FIM":tf.reduce_mean(FIMs,axis=0)},step=epoch)

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
                wandb.log({"GFIM_"+str(i):tf.reduce_mean(tf.gather(FIMs,group_idx),axis=0)},step=(self.epoch*self.max_batch)+self.curr_batch)
                wandb.log({"Loss_"+str(i):tf.reduce_mean(tf.gather(Losses,group_idx),axis=0)},step=(self.epoch*self.max_batch)+self.curr_batch)
        self.curr_batch += 1

class GFIMCallbackBinary(keras.callbacks.Callback):
    def __init__(self, ds, doStep,num_groups,loss_func):
        self.doStep = doStep
        self.ds = ds
        self.num_groups = num_groups
        self.epoch = 0
        self.curr_batch = 0
        self.max_batch = 0
        self.loss_function = loss_func
    

    @tf.function
    def Get_Z_binarycrossentropy(self,items):
        x,y = items
        bs = tf.shape(x)[0]
        
        with tf.GradientTape() as tape:
            #print(text_batch[i])
            y_hat = self.model(x,training=False) #get model output (softmax) [BS x num_classes]
            y = tf.expand_dims(y,axis=1)
            loss = self.loss_function(y,y_hat)
            #selected = tf.squeeze(tf.random.categorical([tf.math.log(y_hat),tf.math.log(1-y_hat)], 1)) #sample from the output [BS x 1]
            #output = tf.gather(y_hat,selected,axis=1,batch_dims=1) #get the output for the selected class [BS x 1]
            #tf.math.log(output) #log the output [BS x 1]
            prob = tf.abs(tf.squeeze(y_hat) - 0.5)+0.5 #prob of maintaining output
            random = tf.random.uniform(tf.shape(prob)) #random number between 0 and 1
            inv_y_hat = 1 - tf.squeeze(y_hat)
            idx_same = tf.where(random < prob, tf.ones_like(prob), tf.zeros_like(prob)) #get the index of the selected output
            idx_swap = tf.where(random < prob, tf.zeros_like(prob), tf.ones_like(prob)) #get the index of the swapped output
            output = tf.squeeze(y_hat)*idx_same + inv_y_hat*idx_swap #get the output based on the index
            #output = tf.math.log(output) #log the output [BS x 1]
            output = tf.expand_dims(output,axis=1)
        
        print(output)
        #print(selected)
        #print(output)

        j = tape.jacobian(output,self.model.trainable_variables)
        layer_sizes = [tf.reduce_sum(tf.size(v)) for v in self.model.trainable_variables] #get the size of each layer
        j = [tf.reshape(j[i],(bs,layer_sizes[i])) for i in range(len(j))] #reshape the gradient to [BS x num_layer_params x layers]
        j = tf.concat(j,axis=1) #concat the gradient over the layers [BS x num_params]
        j = tf.square(j) #square the gradient [BS x num_params]
        j = tf.reduce_sum(j,axis=1) #sum the gradient [ 1]
        j = j/tf.cast(tf.reduce_sum(layer_sizes),tf.float32) #normalize the gradient
        return j, loss

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0:
            self.max_batch = self.curr_batch
        self.epoch = epoch
        if not self.doStep:
            c = 0
            for items in self.ds:
                j, loss = self.Get_Z_binarycrossentropy(items)
                
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
                
                wandb.log({"GFIM_"+str(i):tf.reduce_mean(tf.gather(FIMs,group_idx),axis=0)},step=epoch)
                wandb.log({"Loss_"+str(i):tf.reduce_mean(tf.gather(Losses,group_idx),axis=0)},step=epoch)
            wandb.log({"FIM":tf.reduce_mean(FIMs,axis=0)},step=epoch)

    def on_batch_end(self, batch, logs=None):
        if self.doStep:
            c = 0
            for items in self.ds:
                j, loss = self.Get_Z_binarycrossentropy(items)
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
                wandb.log({"GFIM_"+str(i):tf.reduce_mean(tf.gather(FIMs,group_idx),axis=0)},step=(self.epoch*self.max_batch)+self.curr_batch)
                wandb.log({"Loss_"+str(i):tf.reduce_mean(tf.gather(Losses,group_idx),axis=0)},step=(self.epoch*self.max_batch)+self.curr_batch)
        self.curr_batch += 1



def HeartDesease(ds_root):
    # Load the dataset
    ds_path = os.path.join(ds_root, "heart-disease.csv")
    ds = pd.read_csv(ds_path)

    # Preprocess the dataset
    print(ds.head())
    print(ds["target"].value_counts())
    # normalize the data
    ds = ds.drop_duplicates()
    
    train_ds = ds.sample(frac=0.8, random_state=0)
    test_ds = ds.drop(train_ds.index)
    train_labels = train_ds.pop('target')
    test_labels = test_ds.pop('target')
    train_labels = tf.one_hot(train_labels, 2)
    test_labels = tf.one_hot(test_labels, 2)

    # Create a tf.data.Dataset
    train_ds = tf.data.Dataset.from_tensor_slices((train_ds.values, train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_ds.values, test_labels))

    # Shuffle and batch the datasets
    train_ds = train_ds.shuffle(len(train_ds)).batch(32)
    test_ds = test_ds.batch(32)

    #Define the custom callback to log the learning rate
    class LearningRateLogger(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            lr = self.model.optimizer.learning_rate
            if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
                lr = lr(self.model.optimizer.iterations)
            wandb.log({'learning_rate': lr.numpy()}, step=epoch, commit=False)

    # Define the model
    model = keras.Sequential([
        keras.layers.Dense(13, input_shape=(13,), activation='relu'),
        #keras.layers.Dense(10000, activation='relu'),
        #keras.layers.Dense(100, activation='relu'),
        #keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])
    # Train the model
    wandb.init(project="medical-gfim", name="HeartDesease")
    wandb.config.group = "NGFIM- ModelSize"
    wandb.config.model = "13-2"
    wandb.config.batch_size = 32
    wandb.config.optimizer = 'Adam'
    wandb.config.learning_rate = 0.001
    #decay_lr = tf.keras.optimizers.schedules.ExponentialDecay(0.000001, 1000, 0.9, staircase=False)
    if wandb.config.optimizer == 'SGD':
        optimizer = keras.optimizers.SGD(learning_rate=wandb.config.learning_rate)
    elif wandb.config.optimizer == 'Adam':
        optimizer = keras.optimizers.Adam(learning_rate=wandb.config.learning_rate)
    else:
        print("Invalid optimizer")

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    #nored_loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE)
    nored_loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE)

    FIMCallback = GFIMCallback(train_ds, False, 4, nored_loss_func)
    WandbCallback = wandb.keras.WandbCallback(save_model=False)
    lr_logger = LearningRateLogger()
    model.fit(train_ds, epochs=1000, callbacks=[FIMCallback,lr_logger, WandbCallback],validation_data=test_ds,shuffle=True)
    return

def HeartDeseaseBinary(ds_root):
    # Load the dataset
    ds_path = os.path.join(ds_root, "heart-disease.csv")
    ds = pd.read_csv(ds_path)

    # Preprocess the dataset
    print(ds.head())
    print(ds["target"].value_counts())
    # normalize the data
    ds = ds.drop_duplicates()
    
    train_ds = ds.sample(frac=0.8, random_state=0)
    test_ds = ds.drop(train_ds.index)
    train_labels = train_ds.pop('target')
    test_labels = test_ds.pop('target')

    # Create a tf.data.Dataset
    train_ds = tf.data.Dataset.from_tensor_slices((train_ds.values, train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_ds.values, test_labels))

    # Shuffle and batch the datasets
    train_ds = train_ds.shuffle(len(train_ds)).batch(32)
    test_ds = test_ds.batch(32)

    # Define the model
    model = keras.Sequential([
        keras.layers.Dense(13, input_shape=(13,), activation='relu'),
        keras.layers.Dense(1000, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()


    # Train the model
    wandb.init(project="medical-gfim", name="HeartDesease-Binary")
    wandb.config.epochs = 10
    wandb.config.batch_size = 32
    wandb.config.optimizer = 'SGD'

    #nored_loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE)
    nored_loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE)

    FIMCallback = GFIMCallbackBinary(train_ds, False, 4, nored_loss_func)
    WandbCallback = wandb.keras.WandbCallback(save_model=False)
    model.fit(train_ds, epochs=1000, callbacks=[FIMCallback, WandbCallback],validation_data=test_ds,shuffle=True)
    return





if __name__ == '__main__':
    os.environ['WANDB_API_KEY'] = 'fc2ea89618ca0e1b85a71faee35950a78dd59744'
    wandb.login()

    HeartDesease("/com.docker.devenvironments.code/data/HeartDesease")