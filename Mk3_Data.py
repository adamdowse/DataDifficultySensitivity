#This holds the dataset and relevent functions for that data in a class

import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import numpy as np

class Augmentation():
    def __init__(self,config,pre_aug=False,):
        self.pre_aug = pre_aug    #if true, augment in file location before loading
        self.config = config
        self.aug = self.data_augment()    #augmentation strategy for the data
        # NEED TO SORT AUGMENTS Proberbly remove mem augment and just sequential model for this

    def dir_augment(self):
        #augment the data in the directory
        pass
    
    def data_augment(self):
        print('building augmentation model')
        #augment the data in the data pipeline
        aug = tf.keras.Sequential()
        for aug_name in self.config['augs'].keys():
            if aug_name == 'flip':
                if self.config['augs'][aug_name] == "horizontal":
                    aug.add(tf.keras.layers.RandomFlip('horizontal'))
                elif self.config['augs'][aug_name] == "vertical":
                    aug.add(tf.keras.layers.RandomFlip('vertical'))

            elif aug_name == 'rotate':
                aug.add(tf.keras.layers.experimental.preprocessing.RandomRotation(0.2))

            elif aug_name == 'zoom':
                aug.add(tf.keras.layers.experimental.preprocessing.RandomZoom(0.2))

            elif aug_name == 'crop':
                params = self.config['augs'][aug_name]
                if params != None:
                    #to match torchvision transforms we need to add padding
                    aug.add(tf.keras.layers.RandomCrop(self.config['img_size'][0]-params,self.config['img_size'][1]-params))
                    aug.add(tf.keras.layers.Resizing(self.config['img_size'][0],self.config['img_size'][1]))
                else:
                    print('No crop variables provided, not cropping')

            elif aug_name == 'noise':
                aug.add(tf.keras.layers.GaussianNoise(0.1))
            elif aug_name == 'labelCorr':
                aug.add(tf.keras.layers.experimental.preprocessing.RandomTranslation(0.1,0.1))
            elif aug_name == 'resize':
                aug.add(tf.keras.layers.experimental.preprocessing.Resizing(32,32))
            else:
                print('Augmentation not recognised, skipping...')
        return aug
    
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
    def __init__(self,config,strategy=None,data_dir=None):
        self.config = config
        
        #dataset selector function
        self.get_data = self.get_data_selector()



    


    def get_data_selector(self):
        #This selects the data generator to use
        if self.config['data_name'] == 'mnist':
            return self.get_MNIST
        elif self.config['data_name'] == 'imdb_reviews':
            return self.get_imdb_reviews
        elif self.config['data_name'] == 'newswire':
            return self.get_newswire
        elif self.config['data_name'] == 'speech_commands':
            return self.get_speech_commands
        elif self.config['data_name'] == 'cifar10':
            return self.get_CIFAR10
        elif self.config['data_name'] == 'flowers':
            return self.get_flowers
        else:
            print('Dataset not recognised')
            return None


    def get_CIFAR10(self):
        (input_train, target_train), (input_test, target_test) = tf.keras.datasets.cifar10.load_data()
        self.steps_per_epoch = len(input_train)//self.config['batch_size']

        # Retrieve shape from model configuration and unpack into components
        num_classes = 10

        # Convert scalar targets to categorical ones
        target_train = tensorflow.keras.utils.to_categorical(target_train, num_classes)
        target_test = tensorflow.keras.utils.to_categorical(target_test, num_classes)

        # Data augmentation: perform zero padding on datasets
        paddings = tensorflow.constant([[0, 0,], [4, 4], [4, 4], [0, 0]])
        input_train = tensorflow.pad(input_train, paddings, mode="CONSTANT")

        # Data generator for training data
        train_generator = tensorflow.keras.preprocessing.image.ImageDataGenerator(
            validation_split = config.get("validation_split"),
            horizontal_flip = True,
            rescale = 1./255,
            preprocessing_function = tensorflow.keras.applications.resnet50.preprocess_input
        )

        # Generate training and validation batches
        train_batches = train_generator.flow(input_train, target_train, batch_size=config.get("batch_size"), subset="training")
        validation_batches = train_generator.flow(input_train, target_train, batch_size=config.get("batch_size"), subset="validation")
        train_batches = crop_generator(train_batches, config.get("height"))
        validation_batches = crop_generator(validation_batches, config.get("height"))

        # Data generator for testing data
        test_generator = tensorflow.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function = tensorflow.keras.applications.resnet50.preprocess_input,
            rescale = 1./255)

        # Generate test batches
        test_batches = test_generator.flow(input_test, target_test, batch_size=config.get("batch_size"))

        return train_batches, validation_batches, test_batches
    
    def get_CIFAR10(self):
        #returns the CIFAR10 dataset in tf format as onehot and normalised
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        self.steps_per_epoch = len(x_train)//self.batch_size
        
        #map x to float32 and normalise
        x_train = tf.cast(x_train,tf.float32)
        x_test = tf.cast(x_test,tf.float32)
        if self.split[2] != 0:
            x_val = tf.cast(x_val,tf.float32)

        #map y to one hot
        y_train = tf.one_hot(y_train,10)
        y_test = tf.one_hot(y_test,10)
        if self.split[2] != 0:
            y_val = tf.one_hot(y_val,self.num_classes)
        
        #need to remove the extra dimension
        y_train = tf.squeeze(y_train,axis=1)
        y_test = tf.squeeze(y_test,axis=1)
        if self.split[2] != 0:
            y_val = tf.squeeze(y_val,axis=1)

        #Convert to tf dataset
        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        if self.split[2] != 0:
            val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        else:
            val_data = None
        
        #augment the data
        if self.config['augs'] != None:
            print('Augmenting data')
            train_data = train_data.map(lambda x,y:(self.augmentation.aug(x), y) , num_parallel_calls=tf.data.AUTOTUNE)

        #shuffle and batch data
        train_data = train_data.shuffle(self.train_count,reshuffle_each_iteration=True).batch(self.batch_size)
        test_data = test_data.batch(self.batch_size)
        if self.split[2] != 0:
            val_data = val_data.batch(self.batch_size)
        self.current_train_batch_size = self.batch_size

        return train_data, test_data, val_data

    def get_flowers(self):
        #get flowers dataset from tfds
        (train_data, test_data),data_info = tfds.load('tf_flowers', split=['train[:80%]', 'train[80%:]'], as_supervised=True, with_info=True)
        self.num_classes = data_info.features['label'].num_classes
        self.train_count = data_info.splits['train[:80%]'].num_examples
        self.test_count = data_info.splits['train[80%:]'].num_examples
        self.steps_per_epoch = self.train_count//self.batch_size
        #cast to float and normalise
        train_data = train_data.map(lambda x,y: (tf.cast(x,tf.float32),y))
        test_data = test_data.map(lambda x,y: (tf.cast(x,tf.float32),y))
        #one hot encode the labels
        train_data = train_data.map(lambda x,y: (x,tf.one_hot(y,self.num_classes)))
        test_data = test_data.map(lambda x,y: (x,tf.one_hot(y,self.num_classes)))
        #augs
        def resize(x):
            #min dimention
            #get the image height and width
            height = tf.shape(x)[0]
            width = tf.shape(x)[1]
            #find the smallest dimention
            min_dim = tf.minimum(height,width)
            #crop the image to the smallest dimention
            x = tf.image.random_crop(x,[min_dim,min_dim,3])
            #resize the image to 180
            return tf.image.resize(x,[180,180])
        train_data = train_data.map(lambda x,y: (resize(x), y) , num_parallel_calls=tf.data.AUTOTUNE)
        test_data = test_data.map(lambda x,y: (resize(x), y) , num_parallel_calls=tf.data.AUTOTUNE)
        train_data = train_data.map(lambda x,y:(self.augmentation.aug(x), y) , num_parallel_calls=tf.data.AUTOTUNE)
        #shuffle and batch data
        train_data = train_data.shuffle(self.train_count,reshuffle_each_iteration=True).batch(self.batch_size)
        test_data = test_data.batch(self.batch_size)
        self.current_train_batch_size = self.batch_size
        return train_data, test_data, None



    def build_data(self):
        #build the dataset from source and hold all in memory 
        #Mainly used to pull small test datasets like mnist and cifar10
        
        #Pull from web source
        #Currently uses only default splits
        if self.dataset_name == 'mnist':
            self.train_data,self.test_data,self.val_data = self.get_MNIST()
        elif self.dataset_name == 'imdb_reviews':
            self.train_data,self.test_data,self.val_data = self.get_imdb_reviews()
        elif self.dataset_name == 'newswire':
            self.train_data,self.test_data,self.val_data = self.get_newswire()
        elif self.dataset_name == 'speech_commands':
                self.train_data,self.test_data,self.val_data = self.get_speech_commands()
        elif self.dataset_name == 'cifar10':
                self.train_data,self.test_data,self.val_data = self.get_CIFAR10()
        elif self.dataset_name == 'flowers':
                self.train_data,self.test_data,self.val_data = self.get_flowers()
        else:
            print('Dataset not recognised')
            return None
        
        
        
    def build_iter_ds(self,shuffle=False,bs=None):
        #Shuffle and batch data
        #bs = None means dont update the batch size
        if shuffle:
            self.train_data = self.train_data.shuffle(self.train_count)
        if bs != None:
            if self.iter_batch_size != bs or self.iter_train_data == None:
                self.iter_train_data = self.train_data.unbatch()
                self.iter_train_data = self.iter_train_data.batch(bs)
                self.iter_batch_size = bs
                print('Batch size updated to: ',bs)
        #self.train_batches = self.iter_train_count//bs
        self.iter_train = iter(self.train_data)

    def shuffle_data(self):
        #shuffle the data
        self.train_data = self.train_data.shuffle(self.train_count)
        self.test_data = self.test_data.shuffle(self.test_count)
        if self.split[2] != 0:
            self.val_data = self.val_data.shuffle(self.val_count)

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
        

