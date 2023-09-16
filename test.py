#testing the generator on multi gpus
import os
import tensorflow as tf
import pandas as pd
import numpy as np




def create_data():
    #Need to add function or other file to build the datasets
    return

class HAM10000Data():
    def __init__(self,strategy,data_dir,preaugment_size):
        #requires the file stucture to be
        #data_dir
        #   -HAM10000_metadata.csv (original metadata)
        #   -data
        #       -all images
        #
        #   -C_1000 (example of dataset with 1000 augmented images per class)
        #       -all aug images
        #   -C_1000testmetadata.csv (test metadata)
        #   -C_1000trainmetadata.csv (train metadata)

        self.strategy = strategy
        self.data_dir = data_dir
        self.preaugment_size = preaugment_size

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

        #count the number of images in the train and test data per class
        self.train_class_count = self.trainmetadata['label'].value_counts()
        self.test_class_count = self.testmetadata['label'].value_counts()
        print("Train class count: ",self.train_class_count)
        print("Test class count: ",self.test_class_count)

        #create a master list of all image names and labels
        self.train_img_names = self.trainmetadata['image_id'].tolist()
        self.train_img_labels = self.trainmetadata['label'].tolist()
        self.train_img_labels = tf.keras.utils.to_categorical(self.train_img_labels, num_classes=self.num_classes, dtype='float32') #one hot encode the labels (might need to be int32)
        
        self.test_img_names = self.testmetadata['image_id'].tolist()
        self.test_img_labels = self.testmetadata['label'].tolist()
        self.test_img_labels = tf.keras.utils.to_categorical(self.test_img_labels, num_classes=self.num_classes, dtype='float32') #one hot encode the labels (might need to be int32)

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
        
    
    def reduce_data(self,method):
        #reduce the data to the number of images wanted
        #this will be done by creating a mask for the data
        if method == "all":
            self.train_index_mask = np.array([True]*len(self.train_img_names))
        elif method == "half":
            self.train_index_mask = np.array([True]*len(self.train_img_names))
            index = np.arange(0,len(self.train_img_names))
            np.random.shuffle(index)
            self.train_index_mask[index[:int(len(index)/2)]] = False
        else:
            raise ValueError("Invalid method, please use 'all' or 'half'")


    def init_data(self,bs,train=True,distributed=True):
        #take the data to be used in next epoch and create the data
        if train:
            file_path = self.train_data_dir
            img_names = self.train_img_names[self.train_index_mask]
            img_labels = self.train_img_labels[self.train_index_mask]
        else:
            file_path = self.test_data_dir
            img_names = self.test_img_names
            img_labels = self.test_img_labels

        #find the number of data points
        num_data = len(img_names)
        

        #create the dataset
        def load_img(img_name):
            #load the image
            img = tf.io.read_file(os.path.join(file_path,img_name+".jpg"))
            return img
        x_dataset = tf.data.Dataset.from_tensor_slices(img_names) #this will be the img names to use in training
        x_dataset = x_dataset.map(load_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)#this is now imgs

        y_dataset = tf.data.Dataset.from_tensor_slices(img_labels) #this will be the corrosponding labels

        dataset = tf.data.Dataset.zip((x_dataset, y_dataset))
        dataset = dataset.shuffle(buffer_size=num_data)
        dataset = dataset.batch(bs) #TODO is this to global batch size or per gpu batch size, needs testing
        if distributed:
            dataset = self.strategy.experimental_distribute_dataset(dataset)
        else:
            dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset, num_data



def new_main():

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    data_dir = "/com.docker.devenvironments.code/HAM10000/"
    preaugment_size = 1000
    data = HAM10000Data(strategy,data_dir,preaugment_size)
    test_dataset = data.init_data(10,train=False,distributed=False) #returns a dataset
    train_dataset = data.init_data(12,train=True,distributed=True) #returns a dataset
    iter_dataset = iter(train_dataset)
    for i in range(5): #this needs to be the number of batches including the remainder
        print(iter_dataset.get_next())
    pass




if __name__ == "__main__":
    #main()
    new_main()