#This project aims to connect the inital FIM of a variety of CNN models with datasets.
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import wandb
import os
import Init_Models as im
import random



def get_ds(config):
    #datasetname = "ds_name+_+img_shape" can also be "ds_name+_+img_shape+_+rrandom_aug_percent"
    #if config['dataset'] contains _ then split it
    # if '_' in config['dataset']:
    #     base_ds_name = config['dataset'].split('_')[0]
    #     img_shape = config['dataset'].split('_')[1]
    #     img_shape = tuple([int(x) for x in img_shape[1:-1].split(',')])
    #     if len(config['dataset'].split('_')) > 2:
    #         random_aug_percent = float(config['dataset'].split('_')[2])
    #     else:
    #         random_aug_percent = 0
        

    #     print('random_aug_percent:',random_aug_percent)


    #"ds_name+_+img_shape+
    #_+R(random_img_aug_percent)+
    #_+RL(random_label_aug_percent)+
    #_+C(class_limiter)"
    if '_' in config['dataset']:
        config_array = config['dataset'].split('_')
        base_ds_name = config_array[0]
        img_shape = config_array[1]
        img_shape = tuple([int(x) for x in img_shape[1:-1].split(',')])
        #remove items 0 and 1 from the array and check for other config items
        if len(config_array) > 2:
            config_array = config_array[2:]
            for item in config_array:
                item = item.replace(')','').split('(') #item should now be ["R",random_aug_percent]
                if item[0] == 'R':
                    random_aug_percent = float(item[1])
                elif item[0] == 'RL':
                    random_label_aug_percent = float(item[1])
                elif item[0] == 'C':
                    class_limiter = int(item[1:])
                else:
                    print('Invalid config item')
        #if the values have not been set then set them to None
        if 'random_aug_percent' not in locals():
            random_aug_percent = None
        if 'random_label_aug_percent' not in locals():
            random_label_aug_percent = None
        if 'class_limiter' not in locals():
            class_limiter = None
            
    else:
        base_ds_name = config['dataset']
        if config['dataset'] == 'mnist':
            img_shape = (28,28,1)
        elif config['dataset'] == 'cifar10':
            img_shape = (32,32,3)
        elif config['dataset'] == 'fmnist':
            img_shape = (28,28,1)
        elif config['dataset'] == 'black1c':
            img_shape = (28,28,1)
        elif config['dataset'] == 'black10c':
            img_shape = (28,28,1)
        elif config['dataset'] == 'random':
            img_shape = (28,28,1)
        elif config['dataset'] == 'trandom':
            img_shape = (28,28,1)
        elif config['dataset'] == 'singlecolor':
            img_shape = (28,28,1)
        elif config['dataset'] == 'singlecolorR':
            img_shape = (28,28,1)
        elif config['dataset'] == 'singlecolorC':
            img_shape = (28,28,1)
        elif config['dataset'] == 'singlecolorM':
            img_shape = (28,28,1)
        else:
            print('Invalid Dataset')

    print('base_ds_name:',base_ds_name)
    print('img_shape:',img_shape)
    print('random_aug_percent:',random_aug_percent)
    print('random_label_aug_percent:',random_label_aug_percent)
    print('class_limiter:',class_limiter)


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
        elif base_ds_name == 'black1c':
            x_train = np.zeros((config['FIM_data_count'],28,28,1))
            y_train = np.zeros(config['FIM_data_count'])
            num_classes = 10
            og_img_shape = (28,28,1)
        elif base_ds_name == 'black10c':
            x_train = np.zeros((config['FIM_data_count'],28,28,1))
            y_train = np.random.randint(0,10,config['FIM_data_count'])
            num_classes = 10
            og_img_shape = (28,28,1)
        elif base_ds_name == 'random':
            x_train = np.random.random((config['FIM_data_count'],28,28,1))*255
            y_train = np.random.randint(0,10,config['FIM_data_count'])
            num_classes = 10
            og_img_shape = (28,28,1)
        elif base_ds_name == 'trandom':
            x_train = np.random.random((config['FIM_data_count'],img_shape[0],img_shape[1],img_shape[2]))*255
            y_train = np.random.randint(0,10,config['FIM_data_count'])
            num_classes = 10
            og_img_shape = img_shape
        elif base_ds_name in ['singlecolor','singlecolorR','singlecolorC','singlecolorM']:
            #datasets of a single colour images
            og_img_shape = (1,1,1)
            cols = np.linspace(0,255,10)
            y_train = np.random.randint(0,10,config['FIM_data_count'])
            x_train = np.zeros((config['FIM_data_count'],og_img_shape[0],og_img_shape[1],og_img_shape[2]))
            for i in range(len(x_train)):
                x_train[i] = cols[y_train[i]]

            num_classes = 10
            og_img_shape = (1,1,1)
        else:
            print('Invalid Dataset')
            return

        #limit the number of classes if required
        if class_limiter is not None:
            x_train = x_train[y_train < class_limiter]
            y_train = y_train[y_train < class_limiter]
            num_classes = class_limiter
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

    #this is done to ensure the data in this class is not altered
    if base_ds_name in ['trandom']:
        train_ds = train_ds.map(lambda x,y: (x / 255, y))
        train_ds = train_ds.shuffle(config['FIM_data_count']).batch(1)
        return train_ds,img_shape,num_classes

    #resize the images and normalise
    def resize_map(x,y):
        return tf.image.resize(x,img_shape[:-1]),y
    train_ds= train_ds.map(resize_map)
    #convert to grayscale if required
    if og_img_shape[2] != 1 and img_shape[2] == 1:
        train_ds = train_ds.map(lambda x,y: (tf.image.rgb_to_grayscale(x),y))
    train_ds = train_ds.map(lambda x,y: (x / 255, y))
    
    #augment the data if required
    if random_aug_percent is not None:
        def random_aug(x,y):
            #add noise to the image
            x = x + tf.random.normal(x.shape, mean=0.0, stddev=random_aug_percent)
            #cap the values between 0 and 1
            x = tf.clip_by_value(x,0,1)
            return x,y
        # def random_aug(x,y):
        #     #add constant to the image
        #     x = x + tf.random.normal(x.shape, mean=0.0, stddev=0.1)
        #     x = x + random_aug_percent
        #     return x,y
        train_ds = train_ds.map(random_aug)
    
    #augment the labels if required
    if random_label_aug_percent is not None:
        def random_label_aug(x,y):
            #randomly change the label
            if random.random() < random_label_aug_percent:
                y = tf.random.uniform((),minval=0,maxval=num_classes-1,dtype=tf.int64)
            return x,y
        train_ds = train_ds.map(random_label_aug)
    
    train_ds = train_ds.shuffle(config['FIM_data_count']).batch(1)
    return train_ds,img_shape,num_classes

def init_FIM(config):

    #clear tf session
    tf.keras.backend.clear_session()
    
    train_ds,img_shape,num_classes = get_ds(config)

    #pull model from seperate file
    model = im.get_model(config['model_name'],img_shape,num_classes)

    #compile the model
    optimizer = tf.keras.optimizers.SGD(learning_rate=config['learning_rate'])
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

    def Get_tr_div_norm_Z(item):
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

    def record_GFIM(ds,model,e):
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
        wandb.log({'FIM':mean},step=e)
        return

    def Get_g(item):
        img,label = item
        with tf.GradientTape() as tape:
            y_hat = model(img,training=False)
            y_hat = tf.gather(y_hat,label,axis=1,batch_dims=1)
            y_hat = tf.squeeze(y_hat)
            y_hat = tf.math.log(y_hat)
        g = tape.gradient(y_hat,model.trainable_variables)
        
        layer_sizes = [tf.reduce_sum(tf.size(v)) for v in model.trainable_variables]
        g = [tf.reshape(g[i],(layer_sizes[i])) for i in range(len(g))][-2:] #only take the last two layers
        g = tf.concat(g,axis=0)
        return g #returns a vector of gradients

    def Get_Z_single_L(item):
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
        g = [tf.square(i) for i in g]
        g = [tf.reduce_sum(i) for i in g]
        return g # returns a vector of gradients

    def record_LFIM(ds,model,e):
        data_count = 0
        iter_ds = iter(ds)
        low_lim = config['FIM_data_count']
        for _ in range(low_lim):
            data_count += 1
            if data_count % 500 == 0:
                print(data_count)
            x = Get_Z_single_L(next(iter_ds))
            if data_count == 1:
                mean = [0]*len(x)
            delta = [x - mean for x,mean in zip(x,mean)]
            mean = [d/data_count + m for d,m in zip(delta,mean)]
        for i in range(len(mean)):
            wandb.log({'LFIM_'+str(i):mean[i]},step=e)
        return

    def record_gradient_alignmentandmag(ds,model,e):
        data_count = 0
        alignmean = 0
        magmean = 0
        iter_ds = iter(ds)
        low_lim = config['FIM_data_count']
        for _ in range(low_lim):
            data_count += 1
            if data_count % 500 == 0:
                print(data_count)
            if data_count >1:
                g = Get_g(next(iter_ds))
                #alignment
                alignment = tf.tensordot(g,past_g,1)/(tf.norm(g)*tf.norm(past_g))
                align_d = alignment - alignmean
                alignmean += align_d/(data_count-1)
                #magnitude
                mag = tf.reduce_sum(tf.square(g))
                mag_d = mag - magmean
                magmean += mag_d/(data_count-1)
                past_g = g
            else:
                past_g = Get_g(next(iter_ds))
        wandb.log({'Alignment':alignmean},step=e)
        wandb.log({'SMagnitude':magmean},step=e)

    def Get_single_logit(item):
        img,label = item
        with tf.GradientTape() as tape:
            y_hat = model(img,training=False)
            y_hat = tf.gather(y_hat,label,axis=1,batch_dims=1)
            y_hat = tf.squeeze(y_hat) #softmax output
        return y_hat

    def record_logit_variance(ds,model,e):
        data_count = 0
        sum_o = 0
        sum_o2 = 0
        iter_ds = iter(ds)
        low_lim = config['FIM_data_count']
        for _ in range(low_lim):
            data_count += 1
            if data_count % 500 == 0:
                print(data_count)
            x = Get_single_logit(next(iter_ds))
            sum_o += x
            sum_o2 += x**2
        mean = sum_o/data_count
        wandb.log({'SMVar':(sum_o2 - mean**2)/(data_count-1)},step=e)

    def record_single_image():
        #record the first image to wandb
        img,label = next(iter(train_ds))
        wandb.log({'image':wandb.Image(img[0].numpy())},step=0)
        wandb.log({'label':label[0].numpy()},step=0)

    #record the initial FIM
    if False:
        record_single_image()

    for e in range(config['epochs']):
        print('Epoch:',e)
        if e > 0:
            train_ds = train_ds.unbatch()
            train_ds = train_ds.shuffle(config['FIM_data_count'])
            train_ds = train_ds.batch(config['batch_size'])
            model.fit(train_ds,epochs=1,callbacks=[wandb.keras.WandbCallback(save_model=False)])
            train_ds = train_ds.unbatch().batch(1)
        record_GFIM(train_ds,model,e)
        record_LFIM(train_ds,model,e)
        #record_gradient_alignmentandmag(train_ds,model,e)
        record_logit_variance(train_ds,model,e)
    return 

os.environ['WANDB_API_KEY'] = 'fc2ea89618ca0e1b85a71faee35950a78dd59744'
wandb.login()

#ds names = mnist, cifar10, fmnist, black1c [all labels are 0], black10c [random labels],
#           random [random images of 28,28,1 scaled],
#           trandom [random images of provided size],
#           singlecolor [images of a single colour], singlecolorR [images of a single colour with random noise]
#           singlecolorC [images of a single colour with added constant]

#"R(random_img_aug_percent)_RL(random_label_aug_percent)_C(class_limiter)"
config = {
    'model_name':'Dense2',
    'dataset':'mnist_(5,5,1)',
    'FIM_data_count':1500,
    'epochs':20,
    'learning_rate':0.01,
    'batch_size':8
}
wandb.init(project="Init_LFIM",config=config)

init_FIM(config)



