#This holds the dataset and relevent functions for that data in a class

import tensorflow as tf

class Augmentation():
    def __init__(self,augmentation_setup,pre_aug=False,):
        self.pre_aug = pre_aug    #if true, augment in file location before loading
        self.augmentation_setup = augmentation_setup #[augmentation1,augmentation2,...]

    def dir_augment(self):
        #augment the data in the directory
        pass
    
    def mem_augment(self,ds,num_classes):
        #augment the data in memory
        for aug in self.augmentation_setup:
            aug = aug.lower().split('_')
            if len(aug) > 1:
                var = aug[1:]
                var = [float(i) for i in var]
            aug = aug[0]

            if aug == 'flip':
                ds = ds.map(lambda x,y: (tf.image.flip_left_right(x),y))
            elif aug == 'rotate':
                ds = ds.map(lambda x,y: (tf.image.rot90(x),y))
            elif aug == 'zoom':
                if len(var) == 0:
                    print('No zoom variables provided, using [0.1,0.1]')
                    var = [0.1,0.1]
                ds = ds.map(lambda x,y: (tf.image.random_zoom(x,var[0],var[1]),y))
            elif aug == 'crop':
                if len(var) == 0:
                    print('No crop variables provided, using 0.1')
                    var = [0.1]
                ds = ds.map(lambda x,y: (tf.image.random_crop(x,var[0]),y))
            elif aug == 'noise':
                if len(var) == 0:
                    print('No noise variables provided, using 0.1')
                    var = [0.1]
                def add_noise(x):
                    x = x + tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=var[0], dtype=tf.float32)
                    if len(var) >= 3:
                        x = tf.clip_by_value(x, var[1], var[2])
                ds = ds.map(lambda x,y: (add_noise(x),y))
            elif aug == 'labelCorr':
                if len(var) == 0:
                    print('No labelCorr variables provided, using 0.1')
                    var = [0.1]
                def labelCorr(y):
                    if tf.random.uniform(shape=()) < var[0]:
                        y = tf.random.uniform(shape=(), minval=0, maxval=num_classes, dtype=tf.int32)
                    return y
                ds = ds.map(lambda x,y: (x,labelCorr(y)))
            elif aug == 'resize':
                if len(var) == 0:
                    print('No resize variables provided, using 0.5')
                    var = [0.5]
                ds = ds.map(lambda x,y: (tf.image.resize(x,[var[0],var[0]]),y))
            else:
                print('Augmentation not recognised, skipping...')
        return ds




class Data():
    def __init__(self,dataset_name,batch_size,hold_in_mem=True,split=None,strategy=None,
                data_dir=None,train_augment=None,test_augment=None,val_augment=None,reduced=None,train_count=None,test_count=None,val_count=None):
        if not self.name_setup(dataset_name): return None
        self.dataset_name = dataset_name    #name of the dataset

        #Dataset Modification
        if split == None:
            self.split=[0.8,0.2,0]  #if not None, [train, test,val]
        else:
            self.split = split
        self.train_count = train_count
        self.test_count = test_count
        self.val_count = val_count
        self.reduced = reduced    #percentage to reduce the dataset by if not None
        self.train_augment = train_augment    #augmentation strategy class or None if not needed
        self.test_augment = test_augment    #augmentation strategy class or None if not needed
        self.val_augment = val_augment    #augmentation strategy class or None if not needed

        self.hold_in_mem = hold_in_mem  #if true, load all data into memory
        self.strategy = strategy    #strategy for distributing data
        self.data_dir = data_dir    #directory of the dataset, or None if not needed

        self.batch_size = batch_size    #batch size for the data
        self.current_train_batch_size = 0    #current batch size for the data
        self.current_test_batch_size = 0    #current batch size for the data
        self.current_val_batch_size = None    #current batch size for the data

        self.train_batches = None    #number of batches in the dataset
        self.test_batches = None    #number of batches in the dataset



    


    def name_setup(self,dataset_name):
        #take the dataset name and return the relevant variables
        if dataset_name == 'mnist':
            self.x_type = 'img'
            self.num_classes = 10
            self.input_shape = (28,28,1)
            self.total_data_points = 60000
            return True

        elif dataset_name == 'fashion_mnist':
            self.x_type = 'img'
            self.num_classes = 10
            self.input_shape = (28,28,1)
            self.total_data_points = 60000
            return True

        elif dataset_name == 'cifar10':
            self.x_type = 'img'
            self.num_classes = 10
            self.input_shape = (32,32,3)
            self.total_data_points = 50000
            return True

        elif dataset_name == 'cifar100':
            self.x_type = 'img'
            self.num_classes = 100
            self.input_shape = (32,32,3)
            self.total_data_points = 50000
            return True

        else:
            print('Dataset not recognised')
            return False


    def build_data_in_mem(self):
        #build the dataset from source and hold all in memory 
        #Mainly used to pull small test datasets like mnist and cifar10
        
        #Pull from web source
        #Currently uses only default splits
        if self.dataset_name == 'mnist':
            (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
        else:
            print('Dataset not recognised')
            return None
        
        #Resize data if needed
        if self.train_count != None:
            if len(self.x_train) > self.train_count:
                self.x_train = self.x_train[:self.train_count]
                self.y_train = self.y_train[:self.train_count]
            else:
                print('Train count is larger than dataset size so original size is used')
        else:
            self.train_count = len(self.x_train)
            print('Train count not specified, using full dataset')
        
        if self.test_count != None:
            if len(self.x_test) > self.test_count:
                self.x_test = self.x_test[:self.test_count]
                self.y_test = self.y_test[:self.test_count]
            else:
                print('Test count is larger than dataset size so original size is used')
        else:
            self.test_count = len(self.x_test)
            print('Test count not specified, using full dataset')
        
        if self.val_count != None and self.split[2] != 0:
            if len(self.x_val) > self.val_count:
                self.x_val = self.x_val[:self.val_count]
                self.y_val = self.y_val[:self.val_count]
            else:
                print('Val count is larger than dataset size so original size is used')

        #map y to one hot
        self.y_train = tf.one_hot(self.y_train,self.num_classes)
        self.y_test = tf.one_hot(self.y_test,self.num_classes)

        #Convert to tf dataset
        self.train_data = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        self.test_data = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))
        if self.split[2] != 0:
            self.val_data = tf.data.Dataset.from_tensor_slices((self.x_val, self.y_val))
        
        #Augment data if needed
        if self.train_augment != None:
            self.train_augment.mem_augment(self.train_data,self.num_classes)
        if self.test_augment != None:
            self.test_augment.mem_augment(self.test_data,self.num_classes)
        if self.val_augment != None:
            self.val_augment.mem_augment(self.val_data,self.num_classes)
        
        
        #shuffle and batch data
        self.train_data = self.train_data.shuffle(self.train_count).batch(self.batch_size)
        self.current_train_batch_size = self.batch_size

        self.test_data = self.test_data.shuffle(self.test_count).batch(self.batch_size)
        if self.split[2] != 0:
            self.val_data = self.val_data.shuffle(self.val_count).batch(self.batch_size)
        

    def build_train_iter(self,shuffle=False,bs=None):
        #Shuffle and batch data
        #bs = None means dont update the batch size
        if shuffle:
            self.train_data = self.train_data.shuffle(self.train_count)
        if bs != None:
            if self.current_train_batch_size != bs:
                self.train_data = self.train_data.unbatch()
                self.train_data = self.train_data.batch(bs)
                self.current_train_batch_size = bs
                print('Batch size updated to: ',bs)
        self.train_batches = self.train_count//bs
        self.iter_train = iter(self.train_data)

    def build_test_iter(self,shuffle=False,bs=None):
        #Shuffle and batch data
        #bs = None means dont update the batch size
        if shuffle:
            self.test_data = self.test_data.shuffle(self.test_count)
        if bs != None:
            if self.current_test_batch_size != bs:
                self.test_data = self.test_data.unbatch()
                self.test_data = self.test_data.batch(bs)
                self.current_test_batch_size = bs
        self.test_batches = self.test_count//bs
        self.iter_test = iter(self.test_data)

    def build_val_iter(self,shuffle=False,bs=None):
        if self.split[2] != 0:
            if shuffle:
                self.val_data = self.val_data.shuffle(self.val_count)
            if bs != None:
                if self.current_val_batch_size != bs:
                    self.val_data = self.val_data.unbatch()
                    self.val_data = self.val_data.batch(bs)
                    self.current_val_batch_size = bs
            self.iter_val = iter(self.val_data)


    def get_item(self):
        #return the next item in the dataset
        pass
    
    def get_batch(self,ds_type='train'):
        #return the next batch in the dataset
        if ds_type == 'train':
            return next(self.iter_train)
        elif ds_type == 'test':
            return next(self.iter_test)
        elif ds_type == 'val':
            return next(self.iter_val)
        else:
            print('ds_type not recognised, returning train')
            return next(self.iter_train)
        

