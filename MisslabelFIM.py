#misslabel percentage of the cifar10 dfataset and record the FIM of each subset over training

import numpy as np
import tensorflow as tf

from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import wandb 
import os

class LossFIMCallback(keras.callbacks.Callback):
    def __init__(self, ds, epochRecord,loss_func, limit=500,doNorm=False,prefix=""):
        self.ds = ds
        self.epoch = 0
        self.epochRecord = epochRecord
        self.limit = limit
        self.df = pd.DataFrame(columns=['id'])
        self.loss_function = loss_func
        self.doNorm = doNorm
        self.prefix = prefix

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
        
        if epoch in self.epochRecord:
            print("doing LossFIMRecord")
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
            #add the FIMs to the dataframe
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



def main(epochs, n, bs,opt,lr):
    #dowload dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

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

    #make dataset
    #correct_train = tf.data.Dataset.from_tensor_slices((x_train[y_train[:,1] == 0], y_train[y_train[:,1] == 0][:,0]))
    #mislabeled_train = tf.data.Dataset.from_tensor_slices((x_train[y_train[:,1] == 1], y_train[y_train[:,1] == 1][:,0]))
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)) #[:,0]
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    def map_fn(image, label):
        return image, tf.squeeze(tf.one_hot(tf.cast(label,tf.int32), 10))

    train_dataset = train_dataset.map(map_fn)
    #correct_train = correct_train.map(map_fn)
    #mislabeled_train = mislabeled_train.map(map_fn)
    test_dataset = test_dataset.map(map_fn)

    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(bs)
    #correct_train = correct_train.batch(bs)
    #mislabeled_train = mislabeled_train.batch(bs)
    test_dataset = test_dataset.batch(bs)

    # Get an item from the test_dataset
    #print(next(iter(test_dataset))[1].shape)
    print(next(iter(train_dataset))[1].shape)
    #print(next(iter(correct_train))[1].shape)
    #print(next(iter(mislabeled_train))[1].shape)


    #create model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    if opt == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif opt == "SGD":
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    else:
        print("Invalid optimizer")
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    nored_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy()

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[train_acc_metric])

    #FIMMisslabel = GFIMCallback(mislabeled_train, False, 4, nored_loss_fn, False, "Misslabel")
    #FIMCorrect = GFIMCallback(correct_train, False, 4, nored_loss_fn, False, "Correct")
    #FIMAll = GFIMCallback(train_dataset, False, 8, nored_loss_fn, False, "All")
    LossFIM = LossFIMCallback(train_dataset, [0,20,40,60,80,100], nored_loss_fn, limit=10000,prefix="Normal")
    #corrLossFIM = LossFIMCallback(correct_train, [0,20,40,60,80,100], nored_loss_fn, limit=5000,prefix="RandomCorrect")
    WandbCallback = wandb.keras.WandbCallback(save_model=False)

    #train the model
    model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, callbacks=[LossFIM])


def csv_to_graphs():
    df = pd.read_csv("RandomRandomLossFIM.csv")
    df2 = pd.read_csv("RandomCorrectLossFIM.csv")
    fim_idx = [1,3,5,7,9]
    loss_idx = [2,4,6,8,10]
    epochs = [0,20,40,60,80]
    for i in range(len(fim_idx)):
        plt.scatter(df[str(fim_idx[i])],df[str(loss_idx[i])],s=1,alpha=0.2,color="red")
        plt.scatter(df2[str(fim_idx[i])],df2[str(loss_idx[i])],s=1,alpha=0.2,color="green")
        plt.xlabel("FIM")
        plt.ylabel("Loss")
        plt.title("FIM vs Loss at epoch "+str(epochs[i]))
        #log x axis
        #plt.xscale("log")
        #plt.yscale("log")
        #axis limits
        plt.xlim(0,15000) #(10e-10,10e5)
        plt.ylim(0,10) #(10e-8,10e1)
        
        #add grid lines
        plt.grid()
        plt.savefig("RRandomFIMvsLoss_"+str(epochs[i])+".png")
        #clear plot
        plt.clf()

if __name__ == "__main__":
    csv_to_graphs()
    prnt("done")
    os.environ['WANDB_API_KEY'] = 'fc2ea89618ca0e1b85a71faee35950a78dd59744'
    wandb.login()
    wandb.init(project="MisslabelFIM",name="MisslabelLossFIM")
    wandb.config.epochs = 100
    wandb.config.n = 0.2
    wandb.config.bs = 32
    wandb.config.lr = 0.01
    wandb.config.opt = "SGD"
    main(wandb.config.epochs, wandb.config.n, wandb.config.bs, wandb.config.opt, wandb.config.lr)
