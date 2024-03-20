#This project aims to connect the inital FIM of a variety of CNN models with datasets.
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import wandb
import os
import Init_Models as im



COUNT = 20

Best_BS = 0
Best_LR = 0
Best_Val_Acc = 0


def init_FIM(config):
    global Best_BS
    global Best_LR
    #clear tf session
    tf.keras.backend.clear_session()
    
    #setup data 
    if config['dataset'] == 'mnist':
        #get mnist data from tfds
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        #convert to tfds
        x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.int64)

        #noramlise and build the datasets
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(lambda x, y: (x / 255, y))
        train_ds = train_ds.shuffle(60000).batch(1)
    elif config['dataset'] == 'cifar10':
        #get cifar10 data from tfds
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        #convert to tfds
        x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.int64)

        #noramlise and build the datasets
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(lambda x, y: (x / 255, y))
        train_ds = train_ds.shuffle(60000).batch(1)
    else:
        print('Invalid Dataset')
        return

    #pull model from seperate file
    model = im.get_model(config['model_name'],(28,28,1),10)

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
    'dataset':'mnist',
    'FIM_data_count':5000,
}
wandb.init(project="Init_FIM",config=config)

init_FIM(config)



