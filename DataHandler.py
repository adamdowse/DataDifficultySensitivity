#This will handle the dataset and the dataloader
#functions should include:
#downloading the datasets
#reordering the datasets
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from tensorflow import keras
import time
import random

class DataHandler(tf.keras.utils.Sequence):
    def __init__(self,config):
        self.config = config
        self.epoch_num = 0

        #download the train dataset and prepare it
        train_tfds,self.train_info = self.download_dataset(train=True)
        self.num_classes = self.train_info.features['label'].num_classes
        self.DS_imgs,self.DS_labels,self.DS_loss,self.train_tfds = self.prepare_dataset(train_tfds,1000,misslabel=self.config.misslabel)

        #download the test dataset and prepare it
        self.test_tfds,self.test_info = self.download_dataset(train=False)
        self.test_tfds = self.test_tfds.map(lambda img,label: (tf.cast(img,tf.float32),tf.one_hot(label,self.num_classes)))
        self.test_tfds = self.test_tfds.map(lambda img,label: (tf.cast((img-self.min_val)/(self.max_val-self.min_val),tf.float32),label))
        self.test_tfds = self.test_tfds.batch(self.config.batch_size)

        self.total_train_data_points = self.train_info.splits['train'].num_examples
        self.current_train_data_points = 0
        self.current_train_batch_num = 0



    def __getitem__(self,index,training=False,return_loss=False):
        #return the next batch or data point
        if training:
            self.current_train_batch_num += 1
            self.current_train_data_points += self.config.batch_size

        img = self.DS_imgs[self.indexes[index]]
        label = self.DS_labels[self.indexes[index]]
        loss = self.DS_loss[self.indexes[index]]

        img = tf.cast(img,tf.float32)
        label = tf.one_hot(label,self.num_classes)

        if return_loss:
            return img,label,loss
        else:
            return img,label

    def __len__(self):
        return self.num_batches

    def download_dataset(self,train=True):
        #download the dataset
        t = time.time()
        print('INIT: Using ',self.config.data_percentage*100,"'%' of",self.config.data)
        if train:
            split = 'train[:'+str(int(self.config.data_percentage*100))+'%]'
        else:
            split = 'test[:'+str(int(self.config.data_percentage*100))+'%]'  
        tf_ds = tfds.load(self.config.data,with_info=True,shuffle_files=False,as_supervised=True,split=split,data_dir=self.config.ds_path)
        print('--> Download Time: ',time.time()-t)
        
        return tf_ds

    def prepare_dataset(self,tf_ds,bs=1000,misslabel=0):
        #convert tfds to numpy array
        #add loss to each data point
        #normalize the data
        t = time.time()

        DS_imgs = np.array([img for img,label in tf_ds]) #TODO this could be done in batches
        DS_labels = np.array([label for img,label in tf_ds])
        if misslabel != 0 and misslabel != 1:
            #randomly misslabel some of the data make sure not to mislable the same label twice
            for i in range(int(len(DS_labels)*misslabel)):
                rand_index = random.randint(0,len(DS_labels)-1)
                rand_label = random.randint(0,self.num_classes-1)
                while DS_labels[rand_index] == rand_label:
                    rand_label = random.randint(0,self.num_classes-1)
                DS_labels[rand_index] = rand_label
        elif misslabel == 1:
            #mislable all the data
            for i in range(len(DS_labels)):
                rand_label = random.randint(0,self.num_classes-1)
                DS_labels[i] = rand_label

            
            
        DS_loss = np.zeros(len(DS_imgs))
        print('--> Convert Time: ',time.time()-t)

        #find max and min of img and normalize
        t = time.time()
        self.max_val = np.max(DS_imgs)
        self.min_val = np.min(DS_imgs)
        DS_imgs = (DS_imgs-self.min_val)/(self.max_val-self.min_val)
        print('--> Numpy Time: ',time.time()-t)

        #prepare tf dataset
        t = time.time()
        tf_ds = tf_ds.map(lambda img,label: (tf.cast((img-self.min_val)/(self.max_val-self.min_val),tf.float32),tf.one_hot(label,self.num_classes)))
        print('--> DS min: ',self.min_val,' max: ',self.max_val)
        tf_ds = tf_ds.batch(bs)
        print('--> TFDS Time: ',time.time()-t)
        return DS_imgs,DS_labels,DS_loss,tf_ds
    
    def update_dataset_loss(self,model,tf_ds):
        #update the loss for each data point
        #this is done by running the model on each data point
        t = time.time()
        self.DS_loss = np.array([model.get_items_loss(img,label,training=False) for img,label in tf_ds]).flatten()
        print('--> Loss Update Time: ',time.time()-t)


    def epoch_init(self,model,update=True,method='Vanilla'):
        #run on the start of each epoch
        #updates the loss information in the original dataset
        #applies the method to the mod_dataset
        self.update_indexes_with_method(self.config.batch_size,model,method=method,update=update)
    
    def update_indexes_with_method(self,bs,model,method='Vanilla',update=False,stage=None,num_stages=None):
        #this will update the indexes of the dataset with the method

        if update:
            self.update_dataset_loss(model,self.train_tfds)

        if method == 'Vanilla':
            print('Updating DS Indexes: Applying Vanilla Method')
            t = time.time()
            #create an array of indexes and shuffle it
            self.total_train_data_points = self.train_info.splits['train'].num_examples
            self.indexes = np.array([i * np.ones(bs) for i in range(self.total_train_data_points//bs)]).flatten()
            np.random.shuffle(self.indexes)
            self.indexes = np.array([np.argwhere(self.indexes==i).flatten() for i in range(self.total_train_data_points//bs)]) #this is now a 2d array of indexes for each batch
            print(self.indexes.shape)
            self.num_batches = len(self.indexes)
            print('--> Indexes Time: ',time.time()-t)

        elif method == 'HighLossPercentage':
            print('Updating DS: Applying HighLossPercentage Method')
            #find the loss threshold for the percentage of data points and filter the dataset
            t0 = time.time()
            #sort DS by loss and find the loss threshold
            loss_list = self.DS_loss
            loss_list = np.sort(loss_list)
            self.total_train_data_points = int(self.config.method_param*len(loss_list))
            print('--> Total Data Points: ',self.total_train_data_points)
            loss_threshold = loss_list[self.total_train_data_points]
            #create indexes for each batch by filtering the dataset
            index = np.argwhere(self.DS_loss>=loss_threshold).flatten()
            print(index.shape)
            np.random.shuffle(index)
            self.indexes = np.array([index[i*bs:(i+1)*bs] for i in range((self.total_train_data_points//bs))])
            print(self.indexes.shape)
            self.num_batches = len(self.indexes)
            t3 = time.time()
            print('--> Total Time: ',t3-t0) 

        elif method == 'LowLossPercentage':
            print('Updating DS: Applying LowLossPercentage Method')
            #find the loss threshold for the percentage of data points and filter the dataset
            t0 = time.time()
            #sort DS by loss and find the loss threshold
            loss_list = self.DS_loss
            loss_list = np.sort(loss_list)
            self.total_train_data_points = int(self.config.method_param*len(loss_list))
            print('--> Total Data Points: ',self.total_train_data_points)
            self.loss_threshold = loss_list[-self.total_train_data_points]
            #create indexes for each batch by filtering the dataset
            index = np.argwhere(self.DS_loss<=self.loss_threshold).flatten()
            np.random.shuffle(index)
            self.indexes = np.array([index[i*bs:(i+1)*bs] for i in range(self.total_train_data_points//bs)])
            print(self.indexes.shape)
            self.num_batches = len(self.indexes)
            t3 = time.time()
            print('--> Total Time: ',t3-t0)      

        elif method == 'CL':
            #follow the original CL method
            print('Updating DS: Applying CL Method')
            t0 = time.time()

        elif method == 'Staged':
            #This is used to get a percentage of the dataset at different intervals
            print('Updating DS: Applying Staged Method')
            t0 = time.time()
            loss_list = self.DS_loss
            loss_list = np.sort(loss_list)
            self.total_train_data_points = int(len(loss_list)/num_stages)
            print('--> Total Data Points: ',self.total_train_data_points)
            self.loss_threshold = [loss_list[self.total_train_data_points*stage],loss_list[self.total_train_data_points*(stage+1)-1]]
            #create indexes for each batch by filtering the dataset
            index = np.argwhere((self.DS_loss>=self.loss_threshold[0]) & (self.DS_loss<self.loss_threshold[1])).flatten()
            np.random.shuffle(index)
            self.indexes = np.array([index[i*bs:(i+1)*bs] for i in range(self.total_train_data_points//bs)])
            print(self.indexes.shape)
            self.num_batches = len(self.indexes)
            print('num_batches: ',self.num_batches)
            t3 = time.time()
            print('--> Total Time: ',t3-t0)


    








