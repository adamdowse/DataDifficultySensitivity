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
                aug.add(tf.keras.layers.RandomFlip('horizontal'))
            elif aug_name == 'rotate':
                aug.add(tf.keras.layers.experimental.preprocessing.RandomRotation(0.2))
            elif aug_name == 'zoom':
                aug.add(tf.keras.layers.experimental.preprocessing.RandomZoom(0.2))
            elif aug_name == 'crop':
                aug.add(tf.keras.layers.RandomCrop(32,32)) #padding=4
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
        if not self.name_setup(config['data_name']): return None
        self.dataset_name = config['data_name']    #name of the dataset
        self.dataset_root_dir = "/com.docker.devenvironments.code/data/"

        #Dataset Modification
        if config['data_split'] == None:
            self.split=[0.8,0.2,0]  #if not None, [train, test,val]
        else:
            self.split = config['data_split']
        self.train_count = None #These can be changed to reduce the size based on the split data
        self.test_count = None
        self.val_count = None
        self.reduced = None    #percentage to reduce the dataset by if not None
        if config['augs'] != None:
            self.augmentation = Augmentation(self.config)    #augmentation strategy class or None if not needed

        self.strategy = strategy    #strategy for distributing data
        self.data_dir = data_dir    #directory of the dataset, or None if not needed

        self.batch_size = config['batch_size']    #batch size for the data
        self.current_train_batch_size = 0    #current batch size for the data
        self.current_test_batch_size = 0    #current batch size for the data
        self.current_val_batch_size = None    #current batch size for the data

        self.train_batches = None    #number of batches in the dataset
        self.test_batches = None    #number of batches in the dataset
        self.iter_train_data = None    #iterator for the training data
        self.iter_batch_size = None    #batch size for the iterator



    


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
        
        elif dataset_name == 'imdb_reviews':
            self.x_type = 'text'
            self.num_classes = 2
            self.input_shape = None
            self.total_data_points = 50000
            return True
        elif dataset_name == 'newswire':
            self.x_type = 'text'
            self.num_classes = 46
            self.input_shape = None
            self.total_data_points = 11,228 
            return True
        elif dataset_name == 'speech_commands':
            self.x_type = 'audio'
            self.num_classes = 8
            self.input_shape = None
            self.total_data_points = 8000
            return True

        else:
            print('Dataset not recognised')
            return False

    def get_MNIST(self):
        #returns the mnist dataset in tf format as onehot and normalised
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        #Resize data if needed
        if self.train_count != None:
            if len(x_train) > self.train_count:
                x_train = x_train[:self.train_count]
                y_train = y_train[:self.train_count]
            else:
                print('Train count is larger than dataset size so original size is used')
        else:
            self.train_count = len(x_train)
            print('Train count not specified, using full dataset')
        
        if self.test_count != None:
            if len(x_test) > self.test_count:
                x_test = x_test[:self.test_count]
                y_test = y_test[:self.test_count]
            else:
                print('Test count is larger than dataset size so original size is used')
        else:
            self.test_count = len(x_test)
            print('Test count not specified, using full dataset')
        
        if self.val_count != None and self.split[2] != 0:
            if len(x_val) > self.val_count:
                x_val = x_val[:self.val_count]
                y_val = y_val[:self.val_count]
            else:
                print('Val count is larger than dataset size so original size is used')

        #map x to float32 and normalise
        x_train = tf.cast(x_train,tf.float32)/255
        x_test = tf.cast(x_test,tf.float32)/255
        if self.split[2] != 0:
            x_val = tf.cast(x_val,tf.float32)/255

        #map y to one hot
        y_train = tf.one_hot(y_train,self.num_classes)
        y_test = tf.one_hot(y_test,self.num_classes)
        if self.split[2] != 0:
            y_val = tf.one_hot(y_val,self.num_classes)

        #Convert to tf dataset
        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        if self.split[2] != 0:
            val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        else:
            val_data = None

        #shuffle and batch data
        train_data = train_data.shuffle(self.train_count).batch(self.batch_size)
        test_data = test_data.batch(self.batch_size)
        if self.split[2] != 0:
            val_data = val_data.batch(self.batch_size)
        self.current_train_batch_size = self.batch_size
        return train_data, test_data, val_data

    def get_imdb_reviews(self,max_features=10000,sequence_length=250):
        #Load the data
        start_char = 1
        oov_char = 2
        index_from = 3
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(start_char=start_char, oov_char=oov_char, index_from=index_from, num_words=max_features)
        self.num_classes = 2
        self.train_count = len(x_train)
        self.test_count = len(x_test)
        word_index = tf.keras.datasets.imdb.get_word_index(path="imdb_word_index.json")
        inverted_word_index = dict(
            (i + index_from, word) for (word, i) in word_index.items()
        )
        # Update `inverted_word_index` to include `start_char` and `oov_char`
        inverted_word_index[start_char] = "[START]"
        inverted_word_index[oov_char] = "[OOV]"
        decoded_sequence = " ".join(inverted_word_index[i] for i in x_train[0])
        x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=sequence_length)
        x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=sequence_length)
        #print("Decoded sequence:", decoded_sequence)
        #print("Original sequence:", x_train[0])

        #Create the dataset
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        #shuffle and batch data
        train_ds = train_ds.shuffle(self.train_count).batch(self.batch_size)
        test_ds = test_ds.batch(self.batch_size)
        self.current_train_batch_size = self.batch_size

        #BELOW IS NEEDED FOR TEXT INPUT
        # #sequence to text to simulate text input
        # def sequence_to_text(sequence):
        #     words = [inverted_word_index[i] for i in sequence if i > 2]
        #     text = " ".join(words)
        #     return text
        
        # train_ds = train_ds.map(lambda x, y: (sequence_to_text(x), y))
        # test_ds = test_ds.map(lambda x, y: (sequence_to_text(x), y))

        # print("Text example:", train_ds.take(1))

        # #Vectorize the text
        # def custom_standardization(input_data):
        #     #remove start and end tags
        #     stripped_html = tf.strings.regex_replace(input_data, '[START]', '')
        #     lowercase = tf.strings.lower(input_data)
        #     #stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
        #     return tf.strings.regex_replace(stripped_html,'[%s]' % re.escape(string.punctuation),'')

        # vectorize_layer = layers.TextVectorization(
        #     standardize=custom_standardization,
        #     max_tokens=max_features,
        #     output_mode='int',
        #     output_sequence_length=sequence_length)

        # train_text = train_ds.map(lambda x, y: x)
        # vectorize_layer.adapt(train_text)

        # def vectorize_text(text, label):
        #     text = tf.expand_dims(text, -1)
        #     return vectorize_layer(text), label

        # train_ds = train_ds.map(vectorize_text)
        # test_ds = test_ds.map(vectorize_text)

        return train_ds, test_ds, None

    def get_newswire(self,max_features=10000,sequence_length=250):
        #Load the data
        start_char = 1
        oov_char = 2
        index_from = 3
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.reuters.load_data(start_char=start_char, oov_char=oov_char, index_from=index_from, num_words=max_features)
        self.num_classes = 46
        self.train_count = len(x_train)
        self.test_count = len(x_test)
        word_index = tf.keras.datasets.reuters.get_word_index(path="reuters_word_index.json")
        inverted_word_index = dict(
            (i + index_from, word) for (word, i) in word_index.items()
        )
        # Update `inverted_word_index` to include `start_char` and `oov_char`
        inverted_word_index[start_char] = "[START]"
        inverted_word_index[oov_char] = "[OOV]"
        decoded_sequence = " ".join(inverted_word_index[i] for i in x_train[0])
        x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=sequence_length)
        x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=sequence_length)
        #print("Decoded sequence:", decoded_sequence)
        #print("Original sequence:", x_train[0])

        #convert to onehot
        y_train = tf.one_hot(y_train,self.num_classes)
        y_test = tf.one_hot(y_test,self.num_classes)

        #Create the dataset
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        #shuffle and batch data
        train_ds = train_ds.shuffle(self.train_count).batch(self.batch_size)
        test_ds = test_ds.batch(self.batch_size)
        self.current_train_batch_size = self.batch_size

        return train_ds, test_ds, None

    def get_speech_commands(self):
        #https://www.tensorflow.org/tutorials/audio/simple_audio?_gl=1*1890c69*_up*MQ..*_ga*MTUxMTkwNDM1Ny4xNzE3MDg3MzI4*_ga_W0YLR4190T*MTcxNzA4NzMyOC4xLjAuMTcxNzA4NzMyOC4wLjAuMA..
        name = 'mini_speech_commands'
        
        #/com.docker.devenvironments.code/data/mini_speech_commands
        data_dir = pathlib.Path(self.dataset_root_dir + name)
        if not data_dir.exists():
            tf.keras.utils.get_file(
                'mini_speech_commands.zip',
                origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
                extract=True,
                cache_dir='.',
                cache_subdir='data',
            )
        commands = np.array(tf.io.gfile.listdir(str(data_dir)))
        commands = commands[(commands != 'README.md') & (commands != '.DS_Store')]
        print('Loaded Commands:', commands)

        train_ds, test_ds = tf.keras.utils.audio_dataset_from_directory(
            data_dir,
            validation_split=self.split[1],
            seed=123,
            subset='both',
            batch_size=self.batch_size,
            output_sequence_length=16000, #16k samples
        )
        self.label_names = np.array(train_ds.class_names)
        print('Label Names:', self.label_names)

        #remove the extra dimension as all smaples have 1 channel
        def squeeze(x, y):
            return tf.squeeze(x, axis=-1), y

        train_ds = train_ds.map(squeeze,tf.data.AUTOTUNE)
        test_ds = test_ds.map(squeeze,tf.data.AUTOTUNE)

        def make_spec_ds(ds):
            return ds.map(
                map_func=lambda x, y: (self.get_spectrogram(x), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        
        train_ds = make_spec_ds(train_ds)
        test_ds = make_spec_ds(test_ds)

        train_ds = train_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

        #deal with adapted layers in networks
        self.norm_layer = tf.keras.layers.Normalization()
        self.norm_layer.adapt(train_ds.map(lambda x, _: x))

        return train_ds, test_ds, None

    def get_spectrogram(self,waveform):
        # Convert the waveform to a spectrogram via a STFT.
        spectrogram = tf.signal.stft(
            waveform, frame_length=255, frame_step=128)
        # Obtain the magnitude of the STFT.
        spectrogram = tf.abs(spectrogram)
        # Add a `channels` dimension, so that the spectrogram can be used
        # as image-like input data with convolution layers (which expect
        # shape (`batch_size`, `height`, `width`, `channels`).
        spectrogram = spectrogram[..., tf.newaxis]
        return spectrogram

    def get_CIFAR10(self):
        #returns the CIFAR10 dataset in tf format as onehot and normalised
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        print(y_train[0].shape)

        #Resize data if needed
        if self.train_count != None:
            if len(x_train) > self.train_count:
                x_train = x_train[:self.train_count]
                y_train = y_train[:self.train_count]
            else:
                print('Train count is larger than dataset size so original size is used')
        else:
            self.train_count = len(x_train)
            print('Train count not specified, using full dataset')
        
        if self.test_count != None:
            if len(x_test) > self.test_count:
                x_test = x_test[:self.test_count]
                y_test = y_test[:self.test_count]
            else:
                print('Test count is larger than dataset size so original size is used')
        else:
            self.test_count = len(x_test)
            print('Test count not specified, using full dataset')
        
        if self.val_count != None and self.split[2] != 0:
            if len(x_val) > self.val_count:
                x_val = x_val[:self.val_count]
                y_val = y_val[:self.val_count]
            else:
                print('Val count is larger than dataset size so original size is used')

        #map x to float32 and normalise
        x_train = tf.cast(x_train,tf.float32)/255
        x_test = tf.cast(x_test,tf.float32)/255
        if self.split[2] != 0:
            x_val = tf.cast(x_val,tf.float32)/255

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
        train_data = train_data.shuffle(self.train_count).batch(self.batch_size)
        test_data = test_data.batch(self.batch_size)
        if self.split[2] != 0:
            val_data = val_data.batch(self.batch_size)
        self.current_train_batch_size = self.batch_size

        return train_data, test_data, val_data

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
        

