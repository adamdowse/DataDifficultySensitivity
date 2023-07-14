#This will handle the dataset and the dataloader
#functions should include:
#downloading the datasets
#reordering the datasets
import tensorflow as tf
import numpy as np
import tfds

class DataHandler():
    def __init__(self,config):
        self.current_train_batch_num
        self.train_batches
        self.current_test_batch_num
        self.test_batches
        self.config = config
        self.epoch_num = 0

        #download the dataset and prepare it
        train_tfds,self.train_info = download_dataset(train=True)
        self.num_classes = self.train_info.features['label'].num_classes
        self.test_tfds,test_info = download_dataset(train=False)
        self.test_tfds = self.test_tfds.batch(self.config.batch_size)
        self.train_ds = prepare_dataset(train_tfds)



        def download_dataset(self,train=True):
            #download the dataset
            print('INIT: Using ',self.config.data_percentage*100,"'%' of",self.config.data)
            if train:
                split = 'train[:'+str(int(self.config.data_percentage*100))+'%]'
            else:
                split = 'test[:'+str(int(self.config.data_percentage*100))+'%]'  
            return tfds.load(self.config.data,with_info=True,shuffle_files=False,as_supervised=True,split=split,data_dir=self.config.ds_path)

        def prepare_dataset(self,tfds):
            #convert the dataset to dataloader
            #add loss variable to the dataset
            count = -1
            for img,label in tfds:
                count += 1
                img = tf.cast(img,tf.float32)
                label = tf.one_hot(label,self.num_classes)
                loss = tf.zeros((1))
                if count==0:
                    dataloader = tf.data.Dataset.from_tensors((img,label,loss))
                else:
                    dataloader = dataloader.concatenate(tf.data.Dataset.from_tensors((img,label,loss)))
                
            return dataloader

    def epoch_init(self,model,update=True,apply_method=False):
        #run on the start of each epoch
        self.current_train_batch_num = 0
        self.current_test_batch_num = 0
        self.loss_list = []

        def update_loss_map_fn(img,label,loss):
            #it will return the data point with the loss updated
            loss = model.get_loss(img,label)
            self.loss_list.append(loss)
            return img,label,loss
        
        self.mod_train_ds = self.train_ds.unbatch() 

        if update:
            #This is used to update the loss for each data point
            #calculate loss for each data point
            self.mod_train_ds = self.mod_train_ds.apply(update_loss_map_fn)

            if apply_method:
                if self.config.method == 'Vanilla':
                    #do nothing
                    self.mod_train_ds = self.mod_train_ds

                elif self.config.method == 'HighLossPercentage':
                    #find the loss threshold for the percentage of data points and filter the dataset
                    self.loss_list = np.array(self.loss_list)
                    self.loss_list = np.sort(self.loss_list)
                    self.loss_threshold = self.loss_list[int(self.config.method_param*len(self.loss_list))]
                    self.mod_train_ds = self.mod_train_ds.filter(lambda img,label,loss: loss>self.loss_threshold)
                
                elif self.config.method == 'LowLossPercentage':
                    #find the loss threshold for the percentage of data points and filter the dataset
                    self.loss_list = np.array(self.loss_list)
                    self.loss_list = np.sort(self.loss_list)
                    self.loss_threshold = self.loss_list[int(self.config.method_param*len(self.loss_list))]
                    self.mod_train_ds = self.mod_train_ds.filter(lambda img,label,loss: loss<self.loss_threshold)

                self.mod_train_ds = self.mod_train_ds.shuffle(self.mod_train_ds.cardinality())
                self.mod_train_ds = self.mod_train_ds.batch(self.config.batch_size)

        else:
            #shuffle the dataset and batch
            self.mod_train_ds = self.mod_train_ds.shuffle(self.train_ds.cardinality())
            self.mod_train_ds = self.mod_train_ds.batch(self.config.batch_size)
            
        self.num_batches = self.mod_train_ds.cardinality().numpy()
        self.mod_train_ds = iter(self.mod_train_ds.prefetch(tf.data.experimental.AUTOTUNE))

        
    def get_train_batch(self):
        #return the next batch
        imgs,labels,_ = next(self.mod_train_ds)
        self.current_train_batch_num += 1
        return imgs,labels



