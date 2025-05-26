#load the data for the alpha set of experiments
import tensorflow as tf
import numpy as np
import pandas as pd
import os

# config:{
#     'dataset_name': 'mnist',
#     'normalisation_method': 'minmax',
#     'y_mapping_type': 'none',
#     'batch_size': 32,
#     'shuffle': True,
#     'buffer_size': 1000,
#     'prefetch': True
# }

#Augmentations
#     'augmentations': [
#         {'name': 'flip', 'prob': 0.5},
#         {'name': 'rotate', 'max_angle': 0.1},
#         {'name': 'color_inv', 'prob': 0.5}
#     ],
#     'filtering': {
#         'type': 'class',
#         'class': 0

#TODO dataset load funcitons
def data_pipeline(x,y,data_config):
    #normalise the data
    norm_method = data_config['normalisation_method']
    if norm_method == 'minmax':
        x = (x - x.min()) / (x.max() - x.min())
    elif norm_method == 'div255':
        x = x / 255.0
    elif norm_method == 'none':
        pass
    else:
        raise ValueError(f"Unknown normalisation method: {norm_method}. Supported methods are: minmax, zscore, none.")

    #y mapping
    y_map_types = data_config['y_mapping_type']
    for y_map_type in y_map_types:
        if y_map_type == 'onehot':
            num_classes = np.unique(y).shape[0]
            y = tf.keras.utils.to_categorical(y, num_classes)
        else:
            raise ValueError(f"Unknown y mapping type: {y_map_type}. Supported types are: none, onehot, ordinal.")

    #filtering the data
    if data_config['filtering'] is not None:
        filter_name = data_config['filtering']
        if filter_name['type'] == 'class':
            x = x[y == filter_name['class']]
            y = y[y == filter_name['class']]
        else:
            raise ValueError(f"Unknown filtering type: {filter_name}. Supported types are: class.")

    dataset = tf.data.Dataset.from_tensor_slices((x, y))

    #TODO this should be mapping functions
    if data_config["augmentations"] is not None:
        for aug in data_config['augmentations']:
            if aug['name'] == 'flip':
                dataset = dataset.map(lambda x,y,p=aug["prob"]:aug_flip(x,y,p))
            elif aug['name'] == 'rotate':
                dataset = dataset.map(lambda x,y,a=aug["max_anlge"]:aug_rotate(x,y,a))
            elif aug['name'] == 'color_inv':
                dataset = dataset.map(lambda x,y,p=aug["prob"]:aug_color_inv(x,y,p))
            elif aug["name"] == "scale":
                dataset = dataset.map(lambda x,y: (tf.image.resize(x, (aug["height"], aug["width"])), y))
            elif aug["name"] == "to_color":
                dataset = dataset.map(lambda x,y: (tf.image.grayscale_to_rgb(x), y))
            elif aug["name"] == "fill":
                dataset = dataset.map(lambda x, y: (tf.ones_like(x)*tf.random.uniform(shape=[]),y))
            else:
                raise ValueError(f"Unknown augmentation type: {aug}. Supported types are: flip, rotate.")

    
    #batch the data
    if data_config['batch_size'] > 0:
        dataset = dataset.batch(data_config['batch_size'])
    else:
        pass
    
    #shuffle the data
    if data_config['shuffle']:
        dataset = dataset.shuffle(buffer_size=data_config['buffer_size'])
    else:
        pass
    
    #prefetch the data
    if data_config['prefetch']:
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    else:
        pass
    return dataset

def load_data(data_config):
    ds_name = data_config['dataset_name']
    if ds_name == 'mnist':
        (x_train, y_train), (x_test, y_test) = load_mnist()
    elif ds_name == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = load_fashion_mnist()
    elif ds_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = load_cifar10()
    elif ds_name == 'cifar100':
        (x_train, y_train), (x_test, y_test) = load_cifar100()
    else:
        raise ValueError(f"Unknown dataset name: {ds_name}. Supported datasets are: mnist, fashion_mnist, cifar10, cifar100.")

    if data_config["splits"] == ["train", "test"]:
        train_dataset = data_pipeline(x_train, y_train, data_config)
        test_dataset = data_pipeline(x_test, y_test, data_config)
    elif data_config["splits"] == ["train"]:
        train_dataset = data_pipeline(x_train, y_train, data_config)
        test_dataset = None
    elif data_config["splits"] == ["test"]:
        train_dataset = None
        test_dataset = data_pipeline(x_test, y_test, data_config)
    else:
        raise ValueError(f"Unknown splits: {data_config['splits']}. Supported splits are: train, test.")

    return train_dataset, test_dataset

def aug_flip(x,y,prob=0.5):
    if tf.random.uniform(()) < prob:
        x = tf.image.random_flip_left_right(x)
    return x,y
def aug_rotate(x,y,max_angle=0.1):
    angle = tf.random.uniform((), -max_angle, max_angle)
    x = tf.image.rot90(x, k=angle)
    return x,y
def aug_color_inv(x,y,prob=0.5):
    if tf.random.uniform(shape=[]) < prob:
        x = 1 - x
    return x,y

def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    return (x_train, y_train), (x_test, y_test)
def load_fashion_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    return (x_train, y_train), (x_test, y_test)
def load_cifar10():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    return (x_train, y_train), (x_test, y_test)
def load_cifar100():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    return (x_train, y_train), (x_test, y_test)

class AdditionalValidationSets(tf.keras.callbacks.Callback):
    def __init__(self, validation_sets, verbose=0, batch_size=None):
        """
        :param validation_sets:
        2-tuples (validation_data, validation_set_name)
        :param verbose:
        verbosity mode, 1 or 0
        :param batch_size:
        batch size to be used when evaluating on the additional datasets
        """
        super(AdditionalValidationSets, self).__init__()
        self.validation_sets = validation_sets
        for validation_set in self.validation_sets:
            if len(validation_set) != 2:
                raise ValueError()
        self.verbose = verbose
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):

        # evaluate on the additional validation sets
        for validation_set in self.validation_sets:
            if len(validation_set) == 2:
                validation_data, validation_set_name = validation_set
            else:
                raise ValueError()

            results = self.model.evaluate(validation_data,
                                        verbose=self.verbose,
                                        batch_size=self.batch_size)

            for metric, result in zip(self.model.metrics_names,results):
                valuename = validation_set_name + '_' + metric
                wandb.log({valuename: result}, commit=False)