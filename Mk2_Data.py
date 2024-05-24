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
        
        elif dataset_name == 'imdb_reviews':
            self.x_type = 'text'
            self.num_classes = 2
            self.input_shape = None
            self.total_data_points = 50000
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


    def build_data_in_mem(self):
        #build the dataset from source and hold all in memory 
        #Mainly used to pull small test datasets like mnist and cifar10
        
        #Pull from web source
        #Currently uses only default splits
        match self.dataset_name:
            case 'mnist':
                self.train_data,self.test_data,self.val_data = self.get_MNIST()
            case 'imdb_reviews':
                self.train_data,self.test_data,self.val_data = self.get_imdb_reviews()
            case _:
                print('Dataset not recognised')
                return None
        
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

    def shuffle_data(self):
        #shuffle the data
        self.train_data = self.train_data.shuffle(self.train_count)
        self.test_data = self.test_data.shuffle(self.test_count)
        if self.split[2] != 0:
            self.val_data = self.val_data.shuffle(self.val_count)

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
        

