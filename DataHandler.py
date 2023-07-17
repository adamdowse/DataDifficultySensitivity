#This will handle the dataset and the dataloader
#functions should include:
#downloading the datasets
#reordering the datasets
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from tensorflow import keras
import time

class DataHandler():
    def __init__(self,config):
        self.config = config
        self.epoch_num = 0

        #download the dataset and prepare it
        train_tfds,self.train_info = self.download_dataset(train=True)
        self.num_classes = self.train_info.features['label'].num_classes
        self.test_tfds,test_info = self.download_dataset(train=False)
        self.test_tfds = self.test_tfds.batch(self.config.batch_size)
        self.test_tfds = self.test_tfds.map(lambda img,label: (tf.cast(img,tf.float32),tf.one_hot(label,self.num_classes)))
        self.train_ds = self.prepare_dataset(train_tfds)

        self.total_train_data_points = self.train_info.splits['train'].num_examples
        self.current_train_data_points = 0
        self.current_train_batch_num = 0



    def download_dataset(self,train=True):
        #download the dataset
        print('INIT: Using ',self.config.data_percentage*100,"'%' of",self.config.data)
        if train:
            split = 'train[:'+str(int(self.config.data_percentage*100))+'%]'
        else:
            split = 'test[:'+str(int(self.config.data_percentage*100))+'%]'  
        return tfds.load(self.config.data,with_info=True,shuffle_files=False,as_supervised=True,split=split,data_dir=self.config.ds_path)

    def prepare_dataset(self,tf_ds):
        #convert the dataset to dataloader
        #add loss variable to the dataset
        print('INIT: Preparing Dataset')
        
        count = -1
        #batch the dataset
        b = 1000
        tf_ds = tf_ds.batch(b)
        t = time.time()
        for img,label in tf_ds:
            count += 1
            img = tf.cast(img,tf.float32)
            label = tf.one_hot(label,self.num_classes)
            loss = tf.zeros((b,1))
            if count==0:
                dataloader = tf.data.Dataset.from_tensors((img,label,loss))
            else:
                dataloader = dataloader.concatenate(tf.data.Dataset.from_tensors((img,label,loss)))
        print('INIT: Preparing Dataset Time: ',time.time()-t)
        return dataloader

    def epoch_init(self,model,update=True,apply_method=False):
        #run on the start of each epoch
        self.current_train_data_points = 0

        def update_loss_map_fn(img,label,loss):
            #it will return the data point with the loss updated
            loss = model.get_item_loss(img,label)
            return img,label,loss
        
        def collect_losses(dataset):
            #it will return the list of losses for the dataset
            dataset = dataset.batch(1000)
            #init empyt np array
            loss_list = np.array([])

            for _,_,loss in dataset:
                loss_list = np.append(loss_list,loss.numpy())
            return loss_list.flatten()
        
        self.mod_train_ds = self.train_ds.unbatch() 

        if update:
            #This is used to update the loss for each data point
            #calculate loss for each data point
            t = time.time()
            self.mod_train_ds = self.mod_train_ds.map(update_loss_map_fn)
            print('Epoch INIT: Updating Losses Via Map')
            print('--> Time: ',time.time()-t)
            #losses are collect on the graph so its less intuitive to use but faster

            if apply_method:
                if self.config.method == 'Vanilla':
                    print('Epoch INIT: Applying Vanilla Method')
                    #do nothing
                    self.mod_train_ds = self.mod_train_ds

                elif self.config.method == 'HighLossPercentage':
                    print('Epoch INIT: Applying HighLossPercentage Method')
                    #find the loss threshold for the percentage of data points and filter the dataset
                    t0 = time.time()
                    self.loss_list = collect_losses(self.mod_train_ds)
                    t1 = time.time()
                    print('--> Collecting Losses Time: ',t1-t0)
                    self.loss_list = np.sort(self.loss_list)
                    self.total_train_data_points = int(self.config.method_param*len(self.loss_list))
                    print('--> Total Data Points: ',self.total_train_data_points)
                    self.loss_threshold = self.loss_list[self.total_train_data_points]

                    t2 = time.time()
                    print('--> Sorting Time: ',t2-t1)
                    self.mod_train_ds = self.mod_train_ds.filter(lambda img,label,loss: loss>self.loss_threshold)
                    t3 = time.time()
                    print('--> Filtering Time: ',t3-t2)
                    print('--> Total Time: ',t3-t0)
                    print('--> Updating Total Data Points: ',self.total_train_data_points)
                
                elif self.config.method == 'LowLossPercentage':
                    print('INIT: Applying LowLossPercentage Method')
                    #find the loss threshold for the percentage of data points and filter the dataset
                    self.loss_list = np.array(self.loss_list)
                    self.loss_list = np.sort(self.loss_list)
                    self.loss_threshold = self.loss_list[int(self.config.method_param*len(self.loss_list))]
                    self.mod_train_ds = self.mod_train_ds.filter(lambda img,lab,loss: loss<self.loss_threshold)


        #shuffle the dataset
        print('Epoch INIT: Shuffling and Batching')
        print('--> Total Data Points: ',self.total_train_data_points)
        self.mod_train_ds = self.mod_train_ds.shuffle(self.total_train_data_points)
        #batch the dataset
        self.build_dataset(self.config.batch_size)
        
    
    def build_dataset(self,bs):
        if bs > 0:
            self.built_mod_train_ds = self.mod_train_ds.batch(bs)

        self.built_mod_train_ds = iter(self.built_mod_train_ds.prefetch(tf.data.experimental.AUTOTUNE))

    def get_next(self,training=False):
        #return the next batch or data point
        if training:
            self.current_train_batch_num += 1
            self.current_train_data_points += self.config.batch_size
            print('Batch: ',self.current_train_batch_num,'/',self.total_train_data_points//self.config.batch_size,'Data Points: ',self.current_train_data_points,'/',self.total_train_data_points)

        img,label,_ = next(self.built_mod_train_ds)
        return img,label,_





