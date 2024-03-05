#Recording the CL and its GFIM

import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import Callback
import wandb
import os
import Init_Models as im


class CustomCallback(Callback):
    def __init__(self, FIM_DS, FIM_name, epoch_multiplier=1):
        super().__init__()
        self.FIM_DS = FIM_DS
        self.FIM_name = FIM_name
        self.epoch_multiplier = epoch_multiplier

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
        
        layer_sizes = [tf.reduce_sum(tf.size(v)) for v in self.model.trainable_variables]
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
        low_lim = 2000
        for _ in range(low_lim):
            data_count += 1
            if data_count % 500 == 0:
                print(data_count)
            x = self.__Get_Z_single(next(iter_ds)) #just one replica can be used here
            delta = x - mean 
            mean += delta/(data_count)
        return mean

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        print(f"Finished epoch {epoch}")
        #record the GFIM
        FIM = self.__record_GFIM()
        print(epoch)
        wandb.log({self.FIM_name:FIM},step=(epoch+1)*self.epoch_multiplier)

        
        




def Build_Dataset(combined=True,name="MNIST",aug_percent=0.2):
    if name == "MNIST":
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
                noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=aug_percent, dtype=tf.float32)
                return image + noise
            image = noise(image)
            return image, label

        hard_ds = hard_ds.map(augment)
    
    elif name == "CIFAR10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

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
                noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=aug_percent, dtype=tf.float32)
                return image + noise
            image = noise(image)
            return image, label

        hard_ds = hard_ds.map(augment)

    #save one picture from each dataset
    easy_img = next(iter(easy_ds))[0]
    wandb.log({"easy_img": [wandb.Image(easy_img.numpy())]},step=0)
    hard_img = next(iter(hard_ds))[0]
    wandb.log({"hard_img": [wandb.Image(hard_img.numpy())]},step=0)


    #combine both datasets and shuffle
    if combined:
        combined_ds = easy_ds.concatenate(hard_ds)
        combined_ds = combined_ds.shuffle(50000).batch(32)
        easy_ds = easy_ds.shuffle(25000).batch(32)
        hard_ds = hard_ds.shuffle(25000).batch(32)
        test_ds = test_ds.batch(32)
        return combined_ds,easy_ds,hard_ds,test_ds
    else:
        easy_ds = easy_ds.shuffle(25000).batch(32)
        hard_ds = hard_ds.shuffle(25000).batch(32)
        test_ds = test_ds.batch(32)
        return easy_ds, hard_ds,test_ds
    

def Main(config):
    max_epochs = int(config["max_epochs"])
    lr = float(config["lr"])

    #pull in the model
    if config["dataset"] == "MNIST":
        img_size = (28,28,1)
        num_classes = 10
    elif config["dataset"] == "CIFAR10":
        img_size = (32,32,3)
        num_classes = 10
    else:
        print("Invalid dataset")
        
    model = im.get_model(config["model"],img_size, num_classes)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    wandb_callback = wandb.keras.WandbCallback(save_model=False)

    #pull in a dataset
    if config["type"] == "Combined":
        #Train on the full combined ds
        dataset,easy_ds,hard_ds,test_ds = Build_Dataset(combined=True, name=config["dataset"],aug_percent=float(config["aug_percent"]))
        model.fit(dataset, validation_data=test_ds ,epochs=max_epochs, callbacks=[wandb_callback,CustomCallback(easy_ds,"Easy_FIM",epoch_multiplier=2),CustomCallback(hard_ds,"Hard_FIM",epoch_multiplier=2)])
    elif config["type"] == "Sequential":
        #Train on the easy dataset first, then the hard dataset
        easy_ds, hard_ds,test_ds = Build_Dataset(combined=False, name=config["dataset"],aug_percent=float(config["aug_percent"]))
        model.fit(easy_ds, epochs=max_epochs,validation_data=test_ds, callbacks=[wandb_callback,CustomCallback(easy_ds,"Easy_FIM"),CustomCallback(hard_ds,"Hard_FIM")])
        model.fit(hard_ds, epochs=max_epochs*2,validation_data=test_ds, initial_epoch=max_epochs, callbacks=[wandb_callback,CustomCallback(easy_ds,"Easy_FIM"),CustomCallback(hard_ds,"Hard_FIM")])
    elif config["type"] == "Easy":
        #Train on the easy dataset only
        easy_ds, hard_ds,test_ds = Build_Dataset(combined=False, name=config["dataset"],aug_percent=float(config["aug_percent"]))
        model.fit(easy_ds, epochs=max_epochs*2,validation_data=test_ds, callbacks=[wandb_callback,CustomCallback(easy_ds,"Easy_FIM"),CustomCallback(hard_ds,"Hard_FIM")])
    elif config["type"] == "Hard":
        #Train on the hard dataset only
        easy_ds, hard_ds,test_ds = Build_Dataset(combined=False, name=config["dataset"],aug_percent=float(config["aug_percent"]))
        model.fit(hard_ds, epochs=max_epochs*2,validation_data=test_ds, callbacks=[wandb_callback,CustomCallback(easy_ds,"Easy_FIM"),CustomCallback(hard_ds,"Hard_FIM")])
    elif config["type"] == "Additive":
        #Train on the easy dataset first, then the combined dataset
        dataset,easy_ds,hard_ds,test_ds = Build_Dataset(combined=True, name=config["dataset"],aug_percent=float(config["aug_percent"]))
        model.fit(easy_ds, epochs=max_epochs,validation_data=test_ds, callbacks=[wandb_callback,CustomCallback(easy_ds,"Easy_FIM"),CustomCallback(hard_ds,"Hard_FIM")])
        model.fit(dataset, epochs=max_epochs+(max_epochs//2+1),validation_data=test_ds, initial_epoch=max_epochs, callbacks=[wandb_callback,CustomCallback(easy_ds,"Easy_FIM",epoch_multiplier=2),CustomCallback(hard_ds,"Hard_FIM",epoch_multiplier=2)])
    else:
        print("Invalid method type")
        return


if __name__ == "__main__":
    os.environ['WANDB_API_KEY'] = 'fc2ea89618ca0e1b85a71faee35950a78dd59744'
    wandb.login()
    config = {
        "type":"Hard",
        "model":"CNN3",
        "dataset":"CIFAR10",
        "aug_percent":"0.2",
        "aug_test":"False",
        "lr":"0.1",
        "max_epochs":"30",
    }
    wandb.init(project="CL_FIM",config=config)
    Main(config)
    print("done")

