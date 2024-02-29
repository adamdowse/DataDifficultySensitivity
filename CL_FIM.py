#Recording the CL and its GFIM

import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import Callback
import wandb
import os
import Init_Models as im


class CustomCallback(Callback):
    def __init__(self, FIM_DS, FIM_name):
        super().__init__()
        self.FIM_DS = FIM_DS
        self.FIM_name = FIM_name

    def __Get_Z_single(self,item):
        img,label = item
        with tf.GradientTape() as tape:
            y_hat = self.model(img,training=False) #[0.1,0.8,0.1,ect] this is output of softmax
            output = tf.squeeze(tf.random.categorical(tf.math.log(y_hat), 1)) #[2]
            #output = tf.gather(y_hat,selected,axis=1,batch_dims=1)
            output = tf.gather(y_hat,output,axis=1) #[0.3]
            output = tf.squeeze(output)
            output = tf.math.log(output)
        g = tape.gradient(output,self.model.trainable_variables)#This or Jacobian?
        
        layer_sizes = [tf.reduce_sum(tf.size(v)) for v in model.trainable_variables]
        g = [tf.reshape(g[i],(layer_sizes[i])) for i in range(len(g))] #TODO check that this dosent need to deal with batches
        g = tf.concat(g,axis=0)
        g = tf.square(g)
        g = tf.reduce_sum(g)
        return g

    def __record_GFIM(self):
        print('recording GFIM')
        ds = self.FIM_DS.unbatch()
        ds = ds.batch(1)
        data_count = 0
        mean = 0
        iter_ds = iter(ds)
        low_lim = 1000
        for _ in range(low_lim):
            data_count += 1
            if data_count % 500 == 0:
                print(data_count)
            x = self.__Get_Z_single(next(iter_ds)) #just one replica can be used here
            delta = x - mean 
            mean += delta/(data_count)
        return mean

    def on_epoch_begin(self, epoch, logs=None):
        print(f"Starting epoch {epoch}")

    def on_epoch_end(self, epoch, logs=None):
        print(f"Finished epoch {epoch}")
        #record the GFIM
        FIM = self.__record_GFIM()
        wandb.log({self.FIM_name:FIM},step=epoch)
        
        




def Build_Dataset(combined=True):
    #download the mnist dataset

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    #split the dataset in half
    x_train_easy = x_train[:len(x_train)//2]
    y_train_easy = y_train[:len(y_train)//2]
    x_train_hard = x_train[len(x_train)//2:]
    y_train_hard = y_train[len(y_train)//2:]

    easy_ds = tf.data.Dataset.from_tensor_slices((x_train_easy, y_train_easy))
    hard_ds = tf.data.Dataset.from_tensor_slices((x_train_hard, y_train_hard))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))


    #normalise the datasets
    def normalise(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    hard_ds = hard_ds.map(normalise)
    easy_ds = easy_ds.map(normalise)
    test_ds = test_ds.map(normalise)

    #augment the hard dataset
    def augment(image, label):
        def noise(image):
            noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.2, dtype=tf.float32)
            return image + noise
        image = noise(image)
        return image, label

    hard_ds = hard_ds.map(augment)


    #combine both datasets and shuffle
    if combined:
        combined_ds = easy_ds.concatenate(hard_ds)
        combined_ds = combined_ds.shuffle(50000).batch(32)
        return combined_ds,test_ds
    else:
        easy_ds = easy_ds.shuffle(25000).batch(32)
        hard_ds = hard_ds.shuffle(25000).batch(32)
        return easy_ds, hard_ds,test_ds
    

def Main(combined=True):
    max_epochs = 10
    lr = 0.01


    #pull in the model
    model = im.get_model("CNN3",(28,28,1), 10)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    wandb_callback = wandb.keras.WandbCallback(save_model=False)

    #pull in a dataset
    if combined:
        dataset,test_ds = Build_Dataset(combined=True)
        model.fit(dataset, validation_data=test_ds ,epochs=max_epochs, callbacks=[wandb_callback,CustomCallback(dataset,"FIM")])
    else:
        easy_ds, hard_ds,test_ds = Build_Dataset(combined=False)
        model.fit(easy_ds, epochs=max_epochs, callbacks=[wandb_callback,CustomCallback(easy_ds,"Easy_FIM"),CustomCallback(hard_ds,"Hard_FIM")])
        model.fit(hard_ds, epochs=max_epochs,validation_data=test_ds, initial_epoch=0, callbacks=[wandb_callback,CustomCallback(easy_ds,"Easy_FIM"),CustomCallback(hard_ds,"Hard_FIM")])
        

if __name__ == "__main__":
    os.environ['WANDB_API_KEY'] = 'fc2ea89618ca0e1b85a71faee35950a78dd59744'
    wandb.login()
    wandb.init(project="CL_FIM")
    Main(combined=True)
    print("done")

