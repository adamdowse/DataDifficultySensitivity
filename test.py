#testing the generator on multi gpus
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
import time




def create_data():
    #Need to add function or other file to build the datasets
    #add option method to cull the data to the number of images wanted
    return

class Data():
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
        
    def get_loss(self,model,bs=12):
        #get the loss of the data
        #build dataset
        dataset, num_batches = self.init_data(bs,train=True,distributed=False,shuffle=False)
        #get the loss
        iterator = iter(dataset)
        losses = np.array([])
        for i in range(num_batches):
            batch = next(iterator)
            losses = np.append(losses,model.get_losses(batch[0],batch[1]))
        self.losses = np.array([np.random.rand() for i in range(len(self.train_img_names))])

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
            true_loss_indexes = np.argsort(self.losses)[int(params[0]*len(self.train_index_mask)):int(params[1]*len(self.train_index_mask))]
            self.train_index_mask[true_loss_indexes] = True

        else:
            raise ValueError("Invalid method, please use 'all' or 'half'")

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

        y_dataset = tf.data.Dataset.from_tensor_slices(img_labels) #this will be the corrosponding labels
        y_dataset = y_dataset.map(lambda x: tf.one_hot(x,depth=self.num_classes), num_parallel_calls=tf.data.experimental.AUTOTUNE)#this is now one hot encoded labels

        dataset = tf.data.Dataset.zip((x_dataset, y_dataset))
        dataset = dataset.batch(bs) #TODO is this to global batch size or per gpu batch size, needs testing
        if distributed:
            dataset = self.strategy.experimental_distribute_dataset(dataset)
        else:
            dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset, int(num_data/bs)



class model():
    def __init__(self):
        pass
    def get_losses(self,x,y):
        return np.array([np.random.rand() for i in range(len(x))])

def new_main():

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    data_dir = "/com.docker.devenvironments.code/HAM10000/"
    preaugment_size = 1000
    data = Data(strategy,data_dir,preaugment_size)
    m = model()
    print(data.train_img_names)
    print(data.train_img_labels)
    test_dataset,test_batches = data.init_data(10,train=False,distributed=False) #returns a dataset
    train_dataset,train_batches = data.init_data(12,train=True,distributed=True) #returns a dataset
    print("Test dataset: ",test_dataset)
    print("Train dataset: ",train_dataset)
    print("Test num: ",test_batches)
    print("Train num: ",train_batches)
    data.get_loss(m)
    print(len(data.losses))
    train_dataset = data.reduce_data("loss",[0.1,0.2])
    print(data.train_index_mask)
    train_dataset,train_num = data.init_data(12,train=True,distributed=True) #returns a dataset
    print("Train num: ",train_num)







if __name__ == "__main__":
    #main()
    new_main()