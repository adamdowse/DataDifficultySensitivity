#helper functions and classes for the data of LLMs

import numpy as np
import tensorflow as tf
import wandb


class IMDB_reviews():
    def __init__(self,config):
        #Load the data
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=config.vocab_size)

        #Pad the data
        x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=config.max_length)
        x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=config.max_length)

        #Tokenize the data
        #tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=config.vocab_size,st)
        #tokenizer.fit_on_texts(x_train)
        #x_train = tokenizer.texts_to_sequences(x_train)
        #x_test = tokenizer.texts_to_sequences(x_test)

        #Create the dataset
        self.train_loss = [0]*len(x_train)
        self.num_batches = len(x_train)//config.batch_size
        self.train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(config.batch_size)
        self.test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(config.batch_size)

    def __custom_standardization(input_data):
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
        return tf.strings.regex_replace(stripped_html,
                                        '[%s]' % re.escape(string.punctuation),
                                        '')

    def limit_data(self,dataset,limit_type):
        #only take samples based on condition
        if limit_type == "loss":
            #TODO: get loss for all samples
            #TODO: get max and min loss for the required range
            comb_dataset = tf.data.Dataset.zip((dataset,self.train_loss))
            comb_dataset = comb_dataset.filter(lambda x,y,z: z<lmin and z>lmax)
            return comb_dataset.map(lambda x,y: (x,y)) #TODO: check if its (x,y) or (x,y,z)
        else:
            print("ERROR: limit type not recognised")
            return dataset

    def init_iterators(self,shuffle=False):
        if shuffle:
            comb_dataset = tf.data.Dataset.zip((self.train_ds,self.train_loss))
            comb_dataset = comb_dataset.shuffle(10000)
            self.train_ds,self.train_loss = comb_dataset.map(lambda x, y: (x,y)) #TODO: check if its (x,y) or (x,y,z)

            self.train_iter = iter(self.train_ds)
            self.test_iter = iter(self.test_ds.shuffle(10000))
        else:
            print("ERROR: non shuffle not implemented yet")
        self.train_iter = iter(self.train_ds)
        self.test_iter = iter(self.test_ds)

    def get_train_batch(self):
        try:
            return next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_ds)
            return next(self.train_iter)

    def get_test_batch(self):
        try:
            return next(self.test_iter)
        except StopIteration:
            self.test_iter = iter(self.test_ds)
            return next(self.test_iter)

    def get_train_data(self):
        return self.x_train,self.y_train

    def get_test_data(self):
        return self.x_test,self.y_test

    def get_train_ds(self):
        return self.train_ds

    def get_test_ds(self):
        return self.test_ds

def data_setup(config):
    if config.dataset == "IMDB_reviews":
        return IMDB_reviews(config)