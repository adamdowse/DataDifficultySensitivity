import os
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
import time

#this holds is the class for handling all data related tasks

class Data():
    #this is the class for handling all data related tasks

    #to use all data for an epoch use 
    #   data.reduce_data("all")
    #   dataset, num_batches = data.init_data(bs,train=True,distributed=False,shuffle=True)

    #to use staged data for an epoch use based on loss
    #   data.get_loss(model,bs)
    #   data.reduce_data("loss",[0.0,0.5])
    #   dataset, num_batches = data.init_data(bs,train=True,distributed=False,shuffle=True)

    #load the metadata and create the tf.data datasets that can be distributed
    def __init__(self,strategy,data_dir,preaugment_size,img_size):
        #requires the file stucture to be
        #data_dir
        #   -data
        #       -all images
        #
        #   -C_1000 (example of dataset with 1000 augmented images per class)
        #       -all aug images
        #   -C_1000testmetadata.csv (test metadata)
        #   -C_1000trainmetadata.csv (train metadata)
        #
        #   -C_0 (example of dataset with 0 augmented images per class)(this is the original data)

        self.strategy = strategy
        self.data_name = data_dir.split("/")[-1]
        self.data_dir = data_dir
        self.preaugment_size = preaugment_size
        self.img_size = img_size #[244,244,3]HAM [32,32,3]CIFAR10

        #check if the data is in the correct format and the augmented data wanted exists
        if not self.__check_dirs():
            raise ValueError("Data is not in the correct format, please check the data_dir")

        #load the metadata of train and test data [index,image_id,label]
        self.trainmetadata = pd.read_csv(os.path.join(self.data_dir,"C_"+str(self.preaugment_size)+"trainmetadata.csv")) 
        self.testmetadata = pd.read_csv(os.path.join(self.data_dir,"C_"+str(self.preaugment_size)+"testmetadata.csv"))
        
        #create the data dirs
        self.train_data_dir = os.path.join(self.data_dir,"C_"+str(self.preaugment_size))
        self.test_data_dir = os.path.join(self.data_dir,"data")

        #calculate the number of classes
        self.num_classes = len(self.trainmetadata['label'].unique())
        print("Number of classes: ",self.num_classes)
        self.class_names = self.trainmetadata['label'].unique().tolist()
        print("Class names: ",self.class_names)

        #count the number of images in the train and test data per class
        self.train_class_count = self.trainmetadata['label'].value_counts()
        self.test_class_count = self.testmetadata['label'].value_counts()
        print("Train class count: ",self.train_class_count)
        print("Test class count: ",self.test_class_count)

        #create a master list of all image names and labels
        self.train_img_names = np.array(self.trainmetadata['image_id'].tolist())
        self.train_img_labels = np.array(self.trainmetadata['label'].tolist())
        #convert to int labels
        self.train_img_labels = np.array([self.class_names.index(i) for i in self.train_img_labels])
        
        self.test_img_names = np.array(self.testmetadata['image_id'].tolist())
        self.test_img_labels = np.array(self.testmetadata['label'].tolist())
        #convert to int labels
        self.test_img_labels = np.array([self.class_names.index(i) for i in self.test_img_labels]) 

        #create index mask for train and test data
        self.train_index_mask = np.array([True]*len(self.train_img_names))
        self.test_index_mask = np.array([True]*len(self.test_img_names)) #this should always be true for all the data

    #check if the data is in the correct format
    def __check_dirs(self):
        #returns true if the data is in the correct format
        #check if the data is in the correct format
        #check if the data dir exists
        if not os.path.exists(self.data_dir):
            print("ERROR: Data path does not exist, please check the path")
            return False
        elif not os.path.exists(os.path.join(self.data_dir,"C_"+str(self.preaugment_size))):
            print("ERROR: Augmented data path does not exist, please check the path")
            return False
        #check if the metadata exists
        elif not os.path.exists(os.path.join(self.data_dir,"C_"+str(self.preaugment_size)+"trainmetadata.csv")):
            print("ERROR: Train metadata path does not exist, please check the path")
            return False
        elif not os.path.exists(os.path.join(self.data_dir,"C_"+str(self.preaugment_size)+"testmetadata.csv")):
            print("ERROR: Train metadata path does not exist, please check the path")
            return False
        else:
            return True
        
    #get the loss information for the next epoch
    def get_loss(self,model,bs=12):
        #get the loss of the data
        #build dataset
        self.reduce_data(method='all')
        dataset, num_batches = self.init_data(bs,train=True,distributed=False,shuffle=False)
        #get the loss
        iterator = iter(dataset)
        losses = np.array([])
        for i in range(num_batches):
            batch = next(iterator)
            losses = np.append(losses,model.compute_loss(batch[0],batch[1]))
        self.losses = losses

    #reduce the data to the number of images wanted by creating a mask to limited data used
    def reduce_data(self,method,params=None):
        #reduce the data to the number of images wanted
        #this will be done by creating a mask for the data
        if method == "all":
            self.train_index_mask = np.array([True]*len(self.train_img_names))
        elif method == "half":
            self.train_index_mask = np.array([True]*len(self.train_img_names))
            index = np.arange(0,len(self.train_img_names))
            np.random.shuffle(index)
            self.train_index_mask[index[:int(len(index)/2)]] = False
        elif method == "loss":
            #take the split of data between the params according to the loss
            if params == None:
                raise ValueError("Please specify the params as [lower percentage of data to use,upper percentage of data to use]")
            self.train_index_mask = np.array([False]*len(self.train_img_names))
            print("Total ", len(self.train_index_mask))
            print("percentages ", params)
            print("len losses ", len(self.losses))
            true_loss_indexes = np.argsort(self.losses)[int(params[0]*len(self.train_index_mask)):int(params[1]*len(self.train_index_mask))]
            self.train_index_mask[true_loss_indexes] = True
            print("Number of images used above loss threshold: ",len(true_loss_indexes))
        elif method == "FIM":
            #take the split of data between the params according to the FIM
            if params == None:
                raise ValueError("Please specify the params as boolean array of length num_classes")
            params = params[0]
            if params == "all":
                self.train_index_mask = np.array([True]*len(self.train_img_names))
                return
            self.train_index_mask = np.array([False]*len(self.train_img_names)) #set all to false
            true_loss_indexes = np.argsort(self.losses)                         #sort the losses and get the indexes
            num_groups = len(params)                                            #get the number of groups
            group_size = int(len(true_loss_indexes)/num_groups)                 #get the size of each group
            included_indexes = []                                               #create a list to hold the indexes to include
            for i in range(num_groups):
                if params[i]:
                    included_indexes.append(true_loss_indexes[i*group_size:(i+1)*group_size])
            included_indexes = np.concatenate(included_indexes)
            print("Number of images used above FIM threshold: ",len(included_indexes))
            #true_loss_indexes = np.argsort(self.losses)[int(params[0]*len(self.train_index_mask)):int(params[1]*len(self.train_index_mask))]
            self.train_index_mask[included_indexes] = True

        else:
            raise ValueError("Invalid method, please use 'all' or 'half'")

    #take the data to be used in next epoch and create the tf.data datasets that can be distributed
    def init_data(self,bs,train=True,distributed=True,shuffle=True):
        #take the data to be used in next epoch and create the data
        if train:
            file_path = self.train_data_dir
            img_names = self.train_img_names[self.train_index_mask]
            img_labels = self.train_img_labels[self.train_index_mask]

            #find the number of data points
            num_data = len(img_names)

            if shuffle:
                #shuffle the data (BE CAREFUL OF THIS WHEN COLLECTING DATA)
                self.shuffle_index = np.arange(0,num_data)
                np.random.shuffle(self.shuffle_index)
                img_names = img_names[self.shuffle_index]
                img_labels = img_labels[self.shuffle_index]
        else:
            file_path = self.test_data_dir
            img_names = self.test_img_names
            img_labels = self.test_img_labels

            #find the number of data points
            num_data = len(img_names)

        
        #create the dataset
        def load_img(img_name):
            #load the image
            img = tf.io.read_file(str(file_path)+"/"+img_name+".jpg")
            img = tf.image.decode_jpeg(img, channels=self.img_size[-1])
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img, self.img_size[:-1])
            return img
        x_dataset = tf.data.Dataset.from_tensor_slices(img_names) #this will be the img names to use in training
        x_dataset = x_dataset.map(load_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)#this is now imgs
        if self.data_name == "HAM":
            x_dataset = x_dataset.map(tf.keras.applications.inception_resnet_v2.preprocess_input, num_parallel_calls=tf.data.experimental.AUTOTUNE)#this is now preprocessed imgs

        y_dataset = tf.data.Dataset.from_tensor_slices(img_labels) #this will be the corrosponding labels
        y_dataset = y_dataset.map(lambda x: tf.one_hot(x,depth=self.num_classes), num_parallel_calls=tf.data.experimental.AUTOTUNE)#this is now one hot encoded labels

        dataset = tf.data.Dataset.zip((x_dataset, y_dataset))
        dataset = dataset.batch(bs) #TODO is this to global batch size or per gpu batch size, needs testing
        if distributed:
            options = tf.distribute.InputOptions(
                experimental_fetch_to_device=True,
                experimental_place_dataset_on_device=True,
                experimental_per_replica_buffer_size=10)
            dataset = self.strategy.experimental_distribute_dataset(dataset, 
                                                                    options=options)
        else:
            dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset, int(num_data/bs)