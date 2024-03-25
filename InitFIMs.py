#This project aims to connect the inital FIM of a variety of CNN models with datasets.
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import wandb
import os
import Init_Models as im
import random



def get_ds(config):
    #datasetname = "ds_name+_+img_shape"
    #if config['dataset'] contains _ then split it
    if '_' in config['dataset']:
        base_ds_name = config['dataset'].split('_')[0]
        img_shape = config['dataset'].split('_')[1]
        img_shape = tuple([int(x) for x in img_shape[1:-1].split(',')])
    else:
        base_ds_name = config['dataset']
        if config['dataset'] == 'mnist':
            img_shape = (28,28,1)
        elif config['dataset'] == 'cifar10':
            img_shape = (32,32,3)
        elif config['dataset'] == 'fmnist':
            img_shape = (28,28,1)
        else:
            print('Invalid Dataset')

    print('base_ds_name:',base_ds_name)
    print('img_shape:',img_shape)
    print(tuple(img_shape[:-1]))


    #setup data 
    def get_root_ds(config):
        if base_ds_name == 'mnist':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            num_classes = 10
            og_img_shape = (28,28,1)
        elif base_ds_name == 'cifar10':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            num_classes = 10
            og_img_shape = (32,32,3)
        elif base_ds_name == 'fmnist':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
            num_classes = 10
            og_img_shape = (28,28,1)
        else:
            print('Invalid Dataset')
            return

        bool_list = [True]*config['FIM_data_count'] + [False]*(len(x_train)-config['FIM_data_count'])
        random.shuffle(bool_list)
        x_train = x_train[bool_list]
        y_train = y_train[bool_list]
        #convert to tfds
        x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
        if base_ds_name in ['mnist','fmnist']:
            x_train = tf.expand_dims(x_train, axis=-1)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.int64)

        #build the datasets
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        return train_ds,num_classes,og_img_shape

    #get the root dataset as tf ds
    train_ds,num_classes, og_img_shape = get_root_ds(config)
    def resize_map(x,y):
        return tf.image.resize(x,img_shape[:-1]),y
    #resize the images and normalise
    train_ds= train_ds.map(resize_map)
    if og_img_shape[2] != 1 and img_shape[2] == 1:
        train_ds = train_ds.map(lambda x,y: (tf.image.rgb_to_grayscale(x),y))
    train_ds = train_ds.map(lambda x,y: (x / 255, y))
    train_ds = train_ds.shuffle(config['FIM_data_count']).batch(1)
    return train_ds,img_shape,num_classes

def init_FIM(config):

    #clear tf session
    tf.keras.backend.clear_session()
    
    train_ds,img_shape,num_classes = get_ds(config)

    #pull model from seperate file
    model = im.get_model(config['model_name'],img_shape,num_classes)

    #compile the model
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    def Get_Z_single(item):
        img,label = item
        with tf.GradientTape() as tape:
            y_hat = model(img,training=False) #[0.1,0.8,0.1,ect] this is output of softmax
            output = tf.squeeze(tf.random.categorical(tf.math.log(y_hat), 1)) #[2]
            #output = tf.gather(y_hat,selected,axis=1,batch_dims=1)
            output = tf.gather(y_hat,output,axis=1) #[0.3]
            output = tf.squeeze(output)
            output = tf.math.log(output)
        g = tape.gradient(output,model.trainable_variables)#This or Jacobian?
        
        layer_sizes = [tf.reduce_sum(tf.size(v)) for v in model.trainable_variables]
        g = [tf.reshape(g[i],(layer_sizes[i])) for i in range(len(g))] #TODO check that this dosent need to deal with batches
        g = tf.concat(g,axis=0)
        g = tf.square(g)
        g = tf.reduce_sum(g)
        return g

    def record_GFIM(ds,model):
        print('recording GFIM')
        data_count = 0
        mean = 0
        iter_ds = iter(ds)
        low_lim = config['FIM_data_count']
        for _ in range(low_lim):
            data_count += 1
            if data_count % 500 == 0:
                print(data_count)
            x = Get_Z_single(next(iter_ds)) #just one replica can be used here
            delta = x - mean 
            mean += delta/(data_count)
        wandb.log({'FIM':mean},step=0)
        return

    #record the initial FIM
    record_GFIM(train_ds,model)
    return 

os.environ['WANDB_API_KEY'] = 'fc2ea89618ca0e1b85a71faee35950a78dd59744'
wandb.login()


config = {
    'model_name':'Dense2',
    'dataset':'fmnist_(14,14,1)',
    'FIM_data_count':5000,
}
wandb.init(project="Init_FIM",config=config)

init_FIM(config)



