#This is a repository for the models used in the project 
#it should also contain the hyperparameters for the runs

import tensorflow as tf
import wandb   
from tensorflow import keras
from keras import layers
import math
import time
import numpy as np



class Models():
    def __init__(self,config,dataset_info):
        #this needs to define hyperparams as well as the model
        self.epoch_num = 0
        self.batch_num = 0
        self.config = config
        self.dataset_info = dataset_info
        self.optimizer_init()
        self.metrics_init()
        self.model_init()
        self.loss_func_init()
        self.model_compile()
        self.lr_schedule(0,True)
        self.max_acc = 0
        self.early_stop_count = 0
        
    def optimizer_init(self):
        print('INIT: Optimizer: ',self.config.optimizer)
        #this needs to define the optimizer
        if self.config.optimizer == 'Adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
        elif self.config.optimizer == 'SGD':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.config.lr)
        elif self.config.optimizer == 'Momentum':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.config.lr,momentum=self.config.momentum)
        else:
            print('Optimizer not recognised')    

    def loss_func_init(self):
        print('INIT: Loss: ',self.config.loss_func)
        #this needs to define the loss function
        #TODO add more loss functions

        if self.config.loss_func == 'categorical_crossentropy':
            self.loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=self.output_is_logits,label_smoothing=self.config.label_smoothing)
            self.no_reduction_loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=self.output_is_logits,label_smoothing=self.config.label_smoothing,reduction=tf.keras.losses.Reduction.NONE)
        else:
            print('Loss not recognised')

    def metrics_init(self):
        print('INIT: Metrics')
        self.train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
        self.train_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
        self.train_prec_metric = tf.keras.metrics.Precision(name='train_precision')
        self.train_rec_metric = tf.keras.metrics.Recall(name='train_recall')

        self.test_results = [0.0,0.0,0.0,0.0]
        #self.test_loss_metric = tf.keras.metrics.Mean(name='test_loss')
        #self.test_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
        #self.test_prec_metric = tf.keras.metrics.Precision(name='test_precision')
        #self.test_rec_metric = tf.keras.metrics.Recall(name='test_recall')

    def model_init(self):
        def build_resnet(x,vars,num_classes,REG=0):
            kaiming_normal = keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal')

            def conv3x3(x, out_planes, stride=1, name=None):
                x = layers.ZeroPadding2D(padding=1, name=f'{name}_pad')(x)
                return layers.Conv2D(filters=out_planes, kernel_size=3, strides=stride, use_bias=False, kernel_initializer=kaiming_normal,kernel_regularizer=keras.regularizers.l2(REG), name=name)(x)

            def basic_block(x, planes, stride=1, downsample=None, name=None):
                identity = x

                out = conv3x3(x, planes, stride=stride, name=f'{name}.conv1')
                out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn1')(out)
                out = layers.ReLU(name=f'{name}.relu1')(out)

                out = conv3x3(out, planes, name=f'{name}.conv2')
                out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn2')(out)

                if downsample is not None:
                    for layer in downsample:
                        identity = layer(identity)

                out = layers.Add(name=f'{name}.add')([identity, out])
                out = layers.ReLU(name=f'{name}.relu2')(out)

                return out

            def make_layer(x, planes, blocks, stride=1, name=None):
                downsample = None
                inplanes = x.shape[3]
                if stride != 1 or inplanes != planes:
                    downsample = [
                        layers.Conv2D(filters=planes, kernel_size=1, strides=stride, use_bias=False, kernel_initializer=kaiming_normal,kernel_regularizer=keras.regularizers.l2(REG), name=f'{name}.0.downsample.0'),
                        layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.0.downsample.1'),
                    ]

                x = basic_block(x, planes, stride, downsample, name=f'{name}.0')
                for i in range(1, blocks):
                    x = basic_block(x, planes, name=f'{name}.{i}')

                return x

            def resnet(x, blocks_per_layer, num_classes):
                x = layers.ZeroPadding2D(padding=3, name='conv1_pad')(x)
                x = layers.Conv2D(filters=64, kernel_size=7, strides=2, use_bias=False, kernel_initializer=kaiming_normal,kernel_regularizer=keras.regularizers.l2(REG), name='conv1')(x)
                x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn1')(x)
                x = layers.ReLU(name='relu1')(x)
                x = layers.ZeroPadding2D(padding=1, name='maxpool_pad')(x)
                x = layers.MaxPool2D(pool_size=3, strides=2, name='maxpool')(x)

                x = make_layer(x, 64, blocks_per_layer[0], name='layer1')
                x = make_layer(x, 128, blocks_per_layer[1], stride=2, name='layer2')
                x = make_layer(x, 256, blocks_per_layer[2], stride=2, name='layer3')
                x = make_layer(x, 512, blocks_per_layer[3], stride=2, name='layer4')

                

                x = layers.GlobalAveragePooling2D(name='avgpool')(x)
                initializer = keras.initializers.RandomUniform(-1.0 / math.sqrt(512), 1.0 / math.sqrt(512))
                x = layers.Dense(units=num_classes, kernel_initializer=initializer, bias_initializer=initializer, name='fc')(x)
                x = layers.Softmax(name='softmax')(x)
                return x
            return resnet(x, vars,num_classes)


        print('INIT: Model: ',self.config.model_name)
        if self.config.model_init_type == 'RandNorm':
            initialiser = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=self.config.model_init_seed)
        elif self.config.model_init_type == 'RandUnif':
            initialiser = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=self.config.model_init_seed)
        elif self.config.model_init_type == 'GlorotNorm':
            initialiser = tf.keras.initializers.GlorotNormal(seed=self.config.model_init_seed)
        elif self.config.model_init_type == 'GlorotUnif':
            initialiser = tf.keras.initializers.GlorotUniform(seed=self.config.model_init_seed)
        elif self.config.model_init_type == 'HeNorm':
            initialiser = tf.keras.initializers.HeNormal(seed=self.config.model_init_seed)
        elif self.config.model_init_type == 'HeUnif':
            initialiser = tf.keras.initializers.HeUniform(seed=self.config.model_init_seed)
        else:
            print('Model init type not recognised')

        
        #define the model
        if self.config.model_name == "CNN":
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.dataset_info.features['image'].shape, kernel_initializer=initialiser),
                tf.keras.layers.MaxPool2D(),
                tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.MaxPool2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Dense(self.dataset_info.features['label'].num_classes,activation='softmax', kernel_initializer=initialiser)
            ])
            self.output_is_logits = False
        elif self.config.model_name == "ResNet18":
            #build resnet18 model
            inputs = keras.Input(shape=self.dataset_info.features['image'].shape)
            outputs = build_resnet(inputs,[2,2,2,2],self.dataset_info.features['label'].num_classes,self.config.weight_decay)
            self.model = keras.Model(inputs, outputs)
            self.output_is_logits = False
        
        elif self.config.model_name == "ResNetV1-14":
            #https://www.kaggle.com/code/filippokevin/cifar-10-resnet-14/notebook
            inputs = keras.Input(shape=self.dataset_info.features['image'].shape)
            conv_1 = tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding="same")(inputs)
            conv_b1_1 = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding="same")(conv_1)
            conv_b1_2 = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding="same")(conv_b1_1)
            conv_b1_3 = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding="same")(conv_b1_2)
            sum_1 = tf.keras.layers.Concatenate()([conv_1,conv_b1_3])
            avg_1 = tf.keras.layers.AveragePooling2D(pool_size=(2,2))(sum_1)
            conv_b2_1 = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding="same")(avg_1)
            conv_b2_2 = tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding="same")(conv_b2_1)
            conv_b2_3 = tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding="same")(conv_b2_2)
            sum_2 = tf.keras.layers.Concatenate()([avg_1,conv_b2_3])
            avg_2 = tf.keras.layers.AveragePooling2D(pool_size=(2,2))(sum_2)
            conv_b3_1 = tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3),activation='relu',padding="same")(avg_2)
            conv_b3_2 = tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3),activation='relu',padding="same")(conv_b3_1)
            conv_b3_3 = tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3),activation='relu',padding="same")(conv_b3_2)
            sum_3 = tf.keras.layers.Concatenate()([avg_2,conv_b3_3])
            avg_3 = tf.keras.layers.AveragePooling2D(pool_size=(2,2))(sum_3)
            conv_b4_1 = tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),activation='relu',padding="same")(avg_3)
            conv_b4_2 = tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),activation='relu',padding="same")(conv_b4_1)
            conv_b4_3 = tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),activation='relu',padding="same")(conv_b4_2)
            sum_4 = tf.keras.layers.Concatenate()([avg_3,conv_b4_3])
            avg = tf.keras.layers.AveragePooling2D(pool_size=(2,2))(sum_4)
            flat = tf.keras.layers.Flatten()(avg)#problema <--
            dense1 = tf.keras.layers.Dense(16,activation='relu')(flat)
            dense2 = tf.keras.layers.Dense(10,activation='softmax')(dense1)#maxp
            self.model = tf.keras.models.Model(inputs=inputs,outputs=dense2)
            self.output_is_logits = False

        elif self.config.model_name == "TFCNN":
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=self.dataset_info.features['image'].shape),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64,activation='relu'),
                tf.keras.layers.Dense(self.dataset_info.features['label'].num_classes,activation='softmax')
            ])
            self.output_is_logits = False
        elif self.config.model_name == "ACLCNN":
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32,(3,3),activation='elu',input_shape=self.dataset_info.features['image'].shape, kernel_initializer=initialiser,padding='same'),
                tf.keras.layers.Conv2D(32,(3,3),activation='elu', kernel_initializer=initialiser,padding='same'),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Conv2D(64,(3,3),activation='elu', kernel_initializer=initialiser,padding='same'),
                tf.keras.layers.Conv2D(64,(3,3),activation='elu', kernel_initializer=initialiser,padding='same'),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Conv2D(128,(3,3),activation='elu', kernel_initializer=initialiser,padding='same'),
                tf.keras.layers.Conv2D(128,(3,3),activation='elu', kernel_initializer=initialiser,padding='same'),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Conv2D(256,(2,2),activation='elu', kernel_initializer=initialiser,padding='same'),
                tf.keras.layers.Conv2D(256,(2,2),activation='elu', kernel_initializer=initialiser,padding='same'),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512,activation='elu', kernel_initializer=initialiser),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(self.dataset_info.features['label'].num_classes,activation='softmax', kernel_initializer=initialiser)])
            self.output_is_logits = False
        else:
            print('Model not recognised')

        self.model.build(input_shape=self.dataset_info.features['image'].shape + (1,))

    def model_compile(self):
        self.model.summary()
        self.model.compile(optimizer=self.optimizer,loss=self.loss_func,metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])

    def lr_schedule(self,epoch,init=False):
        #this needs to define the learning rate schedule
        if self.config.lr_decay_type == 'exp':
            self.lr = tf.keras.optimizers.schedules.ExponentialDecay(self.config.lr,decay_steps=self.config.lr_decay_param[0],decay_rate=self.config.lr_decay_param[1],staircase=True)
        elif self.config.lr_decay_type == 'fixed':
            self.lr = self.config.lr
        elif self.config.lr_decay_type == 'cosine':
            self.lr = tf.keras.experimental.CosineDecay(self.config.lr,self.config.lr_decay_param[0])
        elif self.config.lr_decay_type == 'cosine_restarts':
            self.lr = tf.keras.experimental.CosineDecayRestarts(self.config.lr,self.config.lr_decay_param[0])
        else:
            print('Learning rate decay type not recognised')
    
    def epoch_init(self):
        #this is called at the start of each epoch
        #lr set
        self.lr_schedule(self.epoch_num)

        #Reset the metrics at the start of the next epoch
        self.train_loss_metric.reset_states()
        self.train_acc_metric.reset_states()
        self.train_prec_metric.reset_states()
        self.train_rec_metric.reset_states()

        #self.test_loss_metric.reset_states()
        #self.test_acc_metric.reset_states()
        #self.test_prec_metric.reset_states()
        #self.test_rec_metric.reset_states()
        return
    
    def early_stop(self):
        #this needs to define the early stop
        #returns true if early stop is triggered
        #check test accuracy
        if self.test_results[1] > self.max_acc:
            self.max_acc = self.test_results[1]
            self.early_stop_count = 0
        else:
            self.early_stop_count += 1

        if self.early_stop_count >= self.config.early_stop:
            print('Early stop triggered')
            return True
        else:
            return False

    def log_metrics(self):
        wandb.log({'train_loss':self.train_loss_metric.result(),'train_acc':self.train_acc_metric.result(),'train_prec':self.train_prec_metric.result(),'train_rec':self.train_rec_metric.result(),'test_loss':self.test_results[0],'test_acc':self.test_results[1],'test_prec':self.test_results[2],'test_rec':self.test_results[3],'lr':self.optimizer.learning_rate.numpy()},step=self.epoch_num)

    def calc_FIM(self,dataset):
        #this needs to define the FIM
        #calc fim diag
        print('FIM: Calculating FIM')
        t = time.time()
        data_count = 0
        msq = 0
        lower_lim = np.min([self.config.record_FIM_n_data_points,dataset.total_train_data_points])
        for i in range(lower_lim):
            img,_ = dataset.__getitem__(i,training=False,return_loss=False)#returns a batch
            data_count += 1
            #calc sum of squared grads for a data point and class square rooted
            z = self.Get_Z(img)
            if data_count == 1:
                mean = z
            delta = z - mean
            mean += delta / (data_count+1) #Welford_cpp from web
            msq += delta * (z - mean)

        train_FIM = mean
        train_FIM_var = msq/(data_count-1)
        print('--> time: ',time.time()-t)
        return train_FIM,train_FIM_var

        
    @tf.function
    def Get_Z(self,img):
        #returns the z value for a given x and y
        with tf.GradientTape() as tape:
            output = self.model(img,training=False)
            #sample from the output distribution
            output = tf.math.log(output[0,tf.random.categorical(tf.math.log(output), 1)[0][0]])

        grad = tape.gradient(output,self.model.trainable_variables) #all grads 
        #select the weights
        #grads = [g for g in grads if ('Filter' in g.name) or ('MatMul' in g.name)]
        grad = [tf.reshape(g,[-1]) for g in grad] #flatten grads
        grad = tf.concat(grad,0) #concat grads
        grad = tf.math.square(grad) #all grads ^2
        grad = tf.math.reduce_sum(grad) #sum of grads
        return grad

    @tf.function
    def train_step(self,imgs,labels):
        with tf.GradientTape() as tape:
            preds = self.model(imgs,training=True)
            loss = self.loss_func(labels,preds)
        grads = tape.gradient(loss,self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.model.trainable_variables))
        self.train_loss_metric(loss)
        self.train_acc_metric(labels,preds)
        self.train_prec_metric(labels,preds)
        self.train_rec_metric(labels,preds)

    @tf.function
    def get_item_loss(self,img,label,training=False):
        #TODO may be better to not use the self loss func here
        #expand dims
        img = tf.expand_dims(img,0)
        label = tf.expand_dims(label,0)
        preds = self.model(img,training=training)
        loss = self.loss_func(label,preds)
        return loss
    
    @tf.function
    def get_items_loss(self,img,label,training=False):
        #TODO may be better to not use the self loss func here
        #expand dims
        preds = self.model(img,training=training)
        loss = self.no_reduction_loss_func(label,preds)
        return loss

    @tf.function
    def get_batch_loss(self,imgs,labels,training=False):
        preds = self.model(imgs,training=training)
        loss = self.loss_func(labels,preds)
        return loss

