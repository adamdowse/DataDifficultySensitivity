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
import os

class DataHandler(tf.keras.utils.Sequence):
    def __init__(self,config):
        self.config = config
        self.epoch_num = 0

        if self.config.data == 'cifar10':
            #download the train dataset and prepare it
            train_tfds,self.train_info = self.TFdownload_dataset(train=True)
            self.num_classes = self.train_info.features['label'].num_classes
            self.DS_imgs,self.DS_labels,self.DS_loss,self.train_tfds = self.CIFAR10prepare_dataset(train_tfds,1000,misslabel=self.config.misslabel)

            #download the test dataset and prepare it
            self.test_tfds,self.test_info = self.TFdownload_dataset(train=False)
            self.test_tfds = self.test_tfds.map(lambda img,label: (tf.cast(img,tf.float32),tf.one_hot(label,self.num_classes)))
            self.test_tfds = self.test_tfds.map(lambda img,label: (tf.cast((img-self.min_val)/(self.max_val-self.min_val),tf.float32),label))
            self.test_tfds = self.test_tfds.batch(self.config.batch_size)

            #self.total_train_data_points = self.train_info.splits['train'].num_examples


        elif self.config.data == 'HAM10000':
            #dataset should already be downlodaded
            self.DS_imgs,self.DS_labels,self.DS_loss,self.train_tfds,self.test_tfds,total_train_data_points,num_test_data_points,img_shape = self.HAMprepare_dataset(self,misslabel=0)
            #build an info object like the one from tfds
            #self.train_info = ('features':{'image':tfds.features.Image(shape=img_shape),'label':tfds.features.ClassLabel(num_classes=7),},)
            class MyDatasetBuilder():
                def __init__(self):
                    self.name="my_dataset"
                    self.version="1.0.0"
                    self.split_datasets={
                        "train": None,
                        "test": None,
                    }
                    self.splits={"train": self.defsplits(total_train_data_points),
                                "test": self.defsplits(num_test_data_points)}
                    self.features={'image':tfds.features.Image(shape=img_shape),'label':tfds.features.ClassLabel(num_classes=7),}
                    self.config="Test Config"
                    self.description="Test Description"
                    self.release_notes={
                        "1.0.0": "Initial release with numbers up to 5!",
                    }
                class defsplits():
                    def __init__(self,examples):
                        self.num_examples = examples


            self.train_info = MyDatasetBuilder()
            self.num_classes = self.train_info.features['label'].num_classes
            print(self.train_info.splits['train'].num_examples)

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
        if self.config.data == 'cifar10':
            label = tf.one_hot(label,self.num_classes)
            #PROBLEMMM

        if return_loss:
            return img,label,loss
        else:
            return img,label

    def __len__(self):
        return self.num_batches

    def TFdownload_dataset(self,train=True):
        #download the dataset
        t = time.time()
        print('INIT: Using ',self.config.data_percentage*100,"'%' of",self.config.data)
        if train:
            split = 'train[:'+str(int(self.config.data_percentage*100))+'%]'
        else:
            split = 'test[:'+str(int(self.config.data_percentage*100))+'%]'  

        if self.config.data == 'cifar10':
            tf_ds,info = tfds.load(self.config.data,with_info=True,shuffle_files=False,as_supervised=True,split=split,data_dir=self.config.ds_path)
        else:
            print('ERROR: Dataset not found')

        print('--> Download Time: ',time.time()-t)
        return tf_ds,info

    def CIFAR10prepare_dataset(self,tf_ds,bs=1000,misslabel=0,normalize=True):
        #These methods hold whole ds in memory (be aware of this)
        #convert tfds to numpy array
        #add loss to each data point
        #normalize the data
        t = time.time()

        #Define Imgs
        DS_imgs = np.array([img for img,label in tf_ds]) #TODO this could be done in batches

        #Define Labels
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

        #Define Loss
        DS_loss = np.zeros(len(DS_imgs))
        print('--> Convert Time: ',time.time()-t)

        #find max and min of img and normalize
        t = time.time()
        if normalize:
            self.max_val = np.max(DS_imgs)
            self.min_val = np.min(DS_imgs)
            DS_imgs = (DS_imgs-self.min_val)/(self.max_val-self.min_val)
            print('--> Normalize Time: ',time.time()-t)

        #prepare tf dataset
        t = time.time()
        tf_ds = tf_ds.map(lambda img,label: (tf.cast((img-self.min_val)/(self.max_val-self.min_val),tf.float32),tf.one_hot(label,self.num_classes)))
        print('--> DS min: ',self.min_val,' max: ',self.max_val)
        tf_ds = tf_ds.batch(bs)
        print('--> TFDS Time: ',time.time()-t)
        return DS_imgs,DS_labels,DS_loss,tf_ds

    def HAMprepare_dataset(self,bs=1000,misslabel=0):
        #These methods hold whole ds in memory (be aware of this)
        #convert tfds to numpy array
        #add loss to each data point
        #normalize the data
        t = time.time()

        ds = tf.keras.utils.image_dataset_from_directory(directory=os.path.join(self.config.ds_path,'reduced/train'),image_size=(299,299),labels='inferred',label_mode='categorical',batch_size=None,shuffle=False)
        ds = ds.map(lambda img,label: (tf.keras.applications.inception_resnet_v2.preprocess_input(img),label))

        #Define Imgs
        DS_imgs = np.array([img for img,label in ds])
        print(DS_imgs[0].shape)

        #Define Labels
        DS_labels = np.array([label for img,label in ds])
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

        #Define Loss
        DS_loss = np.zeros(len(DS_imgs))
        print('--> Convert Time: ',time.time()-t)

        #prepare tf dataset
        t = time.time()
        print('--> Num Data Points: ',len(DS_imgs))
        print('--> TFDS Time: ',time.time()-t)

        print('--> Test DS')
        t = time.time()
        test_ds = tf.keras.utils.image_dataset_from_directory(directory=os.path.join(self.config.ds_path,'test'),image_size=(299,299),labels='inferred',label_mode='categorical',batch_size=16,shuffle=False)
        test_ds = test_ds.map(lambda img,label: (tf.keras.applications.inception_resnet_v2.preprocess_input(img),label))
        img_shape = test_ds.element_spec[0].shape
        img_shape = list(img_shape)
        img_shape.pop(0)
        img_shape = tuple(img_shape)
        print('--> Img Shape: ',img_shape)
        num_test_data_points = sum(1 for _ in test_ds)
        print('--> Num Test Data Points: ',num_test_data_points)
        test_ds = test_ds.batch(self.config.batch_size)
        print('--> Test DS Time: ',time.time()-t)
        self.config.weighted_train_acc_sample_weight = [[self.config.weighted_train_acc_sample_weight[i]] for i in range(len(self.config.weighted_train_acc_sample_weight))]
        self.config.weighted_train_acc_sample_weight = np.array([self.config.weighted_train_acc_sample_weight for i in range(self.config.batch_size)])
        print('--> Weighted Train Acc Sample Weight shape: ',self.config.weighted_train_acc_sample_weight.shape)

        return DS_imgs,DS_labels,DS_loss,ds,test_ds,len(DS_imgs),num_test_data_points,img_shape


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
            print('--> Lowest loss: ',loss_list[0],' Highest loss: ',loss_list[-1])
            self.total_train_data_points = int(len(loss_list)/num_stages)
            print('--> Total Data Points In Subsection: ',self.total_train_data_points)
            self.loss_threshold = [loss_list[self.total_train_data_points*stage],loss_list[self.total_train_data_points*(stage+1)-1]]
            print('--> Loss Thresholds: ',self.loss_threshold)
            #create indexes of all the datapoints between the thresholds
            index = np.argwhere((self.DS_loss>=self.loss_threshold[0]) & (self.DS_loss<=self.loss_threshold[1])).flatten()
            #shuffle these indexes
            np.random.shuffle(index)
            print("index shape",index.shape)
            print(index)
            #indexes in an array of arrays of size bs
            self.indexes = np.array([index[i*bs:(i+1)*bs] for i in range(self.total_train_data_points//bs)])
            #make sure self.indexes is the correct shape
            print(self.indexes.shape)
            print(self.indexes)
            #if self.indexes.ndim != 1:
            #    print('reshaping indexes happening')
            #    print(self.indexes.shape)
            #    self.indexes = self.indexes[:,0]
            print(self.indexes.shape)
            self.num_batches = len(self.indexes)
            print('num_batches: ',self.num_batches)
            t3 = time.time()
            print('--> Total Time: ',t3-t0)


    








