#This is a repository for the models used in the project 
#it should also contain the hyperparameters for the runs


import tensorflow as tf
import wandb   
from tensorflow import keras
from keras import layers

import math
import time
import numpy as np

import Mk2_Losses as custom_losses
import Mk2_Data as custom_data

#CHANGED MOST TO TF.KERAS MIGHT NEED TO CHANGE BACK TO WORK

class Models():
    def __init__(self,config,strategy,data=None):
        #this needs to define hyperparams as well as the model
        print(config)
        self.data = data
        self.strategy = strategy
        self.num_classes = config['num_classes']
        self.config = config
        self.img_shape = config['img_size'] #could add a batch dimension here
        self.optimizer_init()
        self.metrics_init()
        self.model_init(vars=config['model_vars'])
        self.loss_func_init()
        self.model_compile()
        
        #self.lr_schedule(0,True)
        self.max_acc = 0
        self.early_stop_count = 0
        self.pre_process_func = None
    
    class SAM(tf.Module):
        def __init__(self, base_optim, rho=0.05, lam=0.1):
            self.base_optim = base_optim
            self.rho = rho  # ball size
            self.lam = lam #weight decay coff

        def max_step(self,model,x,y):
            #compute grads at current point and move to the maximum in the ball
            with tf.GradientTape() as tape:
                y_hat = model(x,training=True)
                loss = self.loss_func(y,y_hat)
            gs = tape.gradient(loss, model.trainable_variables)
            grad_norm = tf.linalg.global_norm(gs)
            self.eps = (gs * self.rho) / (grad_norm + 1e-12)
            for self.eps, var in zip(self.eps, vs):
                var.assign_add(self.eps)
            return loss,y_hat
        
        def min_step(self,model,x,y):
            with tf.GradientTape() as tape:
                y_hat = model(x,training=True)
                loss = self.loss_func(y,y_hat)
            gs = tape.gradient(loss, model.trainable_variables)
            #move back to the original point
            for self.eps, var in zip(self.eps, vs):
                var.assign_sub(self.eps)
            #apply normal gradient step
            self.base_optim.apply_gradients(zip(gs, model.trainable_variables))
            return loss

        def train_step(self, x,y):
            #compute the max step
            loss,h_hat = self.max_step(model,x,y)
            self.min_step(model,x,y)
            return loss,y_hat
            

    def optimizer_init(self):
        print('INIT: Optimizer: ',self.config['optimizer'])
        #this needs to define the optimizer
        if self.strategy != None:
            with self.strategy.scope():
                self.lr_schedule(0,init=True)
                if self.config['optimizer'] == 'Adam':
                    #lr decay params = [epsilon] defult = [1e-7]
                    self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['lr'], epsilon=self.config['decay_param'][0],beta_1=0.9, beta_2=0.999, amsgrad=False)
                elif self.config['optimizer'] == 'SGD':
                    self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.config['lr'])
                elif self.config['optimizer'] == 'Momentum':
                    self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.config['lr'],momentum=self.config.momentum)
                else:
                    print('Optimizer not recognised')   
        else:
            self.lr_schedule(0,init=True)
            if self.config['optimizer'] == 'Adam':
                #lr decay params = [epsilon] defult = [1e-7]
                self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['lr'])
                #self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['lr'], epsilon=self.config['decay_param'][0],beta_1=0.9, beta_2=0.999, amsgrad=False)
            elif self.config['optimizer'] == 'SGD':
                self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.config['lr'])
            elif self.config['optimizer'] == 'Momentum':
                self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.config['lr'],momentum=self.config.momentum)
            elif self.config['optimizer'] == 'SAM_SGD':
                self.optimizer = self.SAM(tf.keras.optimizers.SGD(learning_rate=self.config['lr']),rho=0.05,lam=0.1)
            else:
                print('Optimizer not recognised')   

    def loss_func_init(self):
        print('INIT: Loss: ',self.config['loss_func'])
        #this needs to define the loss function
        #TODO add more loss functions
        match self.config['loss_func']:
            case 'categorical_crossentropy' :
                self.loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=self.output_is_logits)
            case 'binary_crossentropy' :
                self.loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=self.output_is_logits)
            case 'sparse_categorical_crossentropy' :
                self.loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=self.output_is_logits)
        
        # if self.strategy != None:
        #     with self.strategy.scope():
        #         if self.config['loss_func'] == 'categorical_crossentropy' and self.config['acc_sample_weight'] == None:
        #             #self.loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=self.output_is_logits,label_smoothing=self.config['label_smoothing'])
        #             self.loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=self.output_is_logits)
        #             self.loss_func_nored = tf.keras.losses.CategoricalCrossentropy(from_logits=self.output_is_logits,reduction=tf.keras.losses.Reduction.NONE)
        #         elif self.config['loss_func'] == 'categorical_crossentropy':
        #             #weighted categorical crossentropy
        #             #self.loss_func = WeightedCategoricalCrossentropy(self.config['acc_sample_weight'],reduction=tf.keras.losses.Reduction.AUTO,label_smoothing=self.config['label_smoothing'],from_logits=self.output_is_logits)
        #             self.loss_func = custom_losses.WeightedCategoricalCrossentropy(self.config['acc_sample_weight'],reduction=tf.keras.losses.Reduction.NONE,label_smoothing=self.config['label_smoothing'],from_logits=self.output_is_logits)
        #         else:
        #             print('Loss not recognised')
        # else:
        #     if self.config['loss_func'] == 'categorical_crossentropy' and self.config['acc_sample_weight'] == None:
        #         #self.loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=self.output_is_logits,label_smoothing=self.config['label_smoothing'])
        #         self.loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=self.output_is_logits,label_smoothing=self.config['label_smoothing'])
        #         self.loss_func_nored = tf.keras.losses.CategoricalCrossentropy(from_logits=self.output_is_logits,reduction=tf.keras.losses.Reduction.NONE)
        #     elif self.config['loss_func'] == 'categorical_crossentropy':
        #         #weighted categorical crossentropy
        #         #self.loss_func = WeightedCategoricalCrossentropy(self.config['acc_sample_weight'],reduction=tf.keras.losses.Reduction.AUTO,label_smoothing=self.config['label_smoothing'],from_logits=self.output_is_logits)
        #         self.loss_func = custom_losses.WeightedCategoricalCrossentropy(self.config['acc_sample_weight'],reduction=tf.keras.losses.Reduction.NONE,label_smoothing=self.config['label_smoothing'],from_logits=self.output_is_logits)
        #     else:
        #         print('Loss not recognised')


    def metrics_init(self):
        print('INIT: Metrics')
        #Store for metrics calculated during training
        match self.config['loss_func']:
            case 'categorical_crossentropy' :
                metrics = tf.metrics.CategoricalAccuracy()
            case 'binary_crossentropy' :
                metrics = tf.metrics.BinaryAccuracy(threshold=0.0)
            case 'sparse_categorical_crossentropy' :
                metrics = tf.metrics.SparseCategoricalAccuracy()
            case _:
                print('Loss not recognised for accuracy metric')
        return metrics
        # if self.strategy != None:
        #     with self.strategy.scope():
        #         self.train_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
        #         self.train_prec_metric = tf.keras.metrics.Precision(name='train_precision')
        #         self.train_rec_metric = tf.keras.metrics.Recall(name='train_recall')

        #         self.test_loss_metric = tf.keras.metrics.Mean(name='test_loss')
        #         self.test_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
        #         self.test_prec_metric = tf.keras.metrics.Precision(name='test_precision')
        #         self.test_rec_metric = tf.keras.metrics.Recall(name='test_recall')
        # else:
        #     self.train_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
        #     self.train_prec_metric = tf.keras.metrics.Precision(name='train_precision')
        #     self.train_rec_metric = tf.keras.metrics.Recall(name='train_recall')

        #     self.test_loss_metric = tf.keras.metrics.Mean(name='test_loss')
        #     self.test_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
        #     self.test_prec_metric = tf.keras.metrics.Precision(name='test_precision')
        #     self.test_rec_metric = tf.keras.metrics.Recall(name='test_recall')


    def model_init(self,vars=None):
        #this needs to define the model
        def build_resnet(x,vars,num_classes,REG=0):
            kaiming_normal = keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal')

            def conv3x3(x, out_planes, stride=1, name=None):
                x = tf.keras.layers.ZeroPadding2D(padding=1, name=f'{name}_pad')(x)
                return tf.keras.layers.Conv2D(filters=out_planes, kernel_size=3, strides=stride, use_bias=False, kernel_initializer=kaiming_normal,kernel_regularizer=keras.regularizers.l2(REG), name=name)(x)

            def basic_block(x, planes, stride=1, downsample=None, name=None):
                identity = x

                out = conv3x3(x, planes, stride=stride, name=f'{name}.conv1')
                out = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn1')(out)
                out = tf.keras.layers.ReLU(name=f'{name}.relu1')(out)

                out = conv3x3(out, planes, name=f'{name}.conv2')
                out = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn2')(out)

                if downsample is not None:
                    for layer in downsample:
                        identity = layer(identity)

                out = tf.keras.layers.Add(name=f'{name}.add')([identity, out])
                out = tf.keras.layers.ReLU(name=f'{name}.relu2')(out)

                return out

            def make_layer(x, planes, blocks, stride=1, name=None):
                downsample = None
                inplanes = x.shape[3]
                if stride != 1 or inplanes != planes:
                    downsample = [
                        tf.keras.layers.Conv2D(filters=planes, kernel_size=1, strides=stride, use_bias=False, kernel_initializer=kaiming_normal,kernel_regularizer=keras.regularizers.l2(REG), name=f'{name}.0.downsample.0'),
                        tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.0.downsample.1'),
                    ]

                x = basic_block(x, planes, stride, downsample, name=f'{name}.0')
                for i in range(1, blocks):
                    x = basic_block(x, planes, name=f'{name}.{i}')

                return x

            def resnet(x, blocks_per_layer, num_classes):
                x = tf.keras.layers.ZeroPadding2D(padding=3, name='conv1_pad')(x)
                x = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, use_bias=False, kernel_initializer=kaiming_normal,kernel_regularizer=keras.regularizers.l2(REG), name='conv1')(x)
                x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn1')(x)
                x = tf.keras.layers.ReLU(name='relu1')(x)
                x = tf.keras.layers.ZeroPadding2D(padding=1, name='maxpool_pad')(x)
                x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, name='maxpool')(x)

                x = make_layer(x, 64, blocks_per_layer[0], name='layer1')
                x = make_layer(x, 128, blocks_per_layer[1], stride=2, name='layer2')
                x = make_layer(x, 256, blocks_per_layer[2], stride=2, name='layer3')
                x = make_layer(x, 512, blocks_per_layer[3], stride=2, name='layer4')

                

                x = tf.keras.layers.GlobalAveragePooling2D(name='avgpool')(x)
                initializer = tf.keras.initializers.RandomUniform(-1.0 / math.sqrt(512), 1.0 / math.sqrt(512))
                x = tf.keras.layers.Dense(units=num_classes, kernel_initializer=initializer, bias_initializer=initializer, name='fc')(x)
                x = tf.keras.layers.Softmax(name='softmax')(x)
                return x
            return resnet(x, vars,num_classes)

        class SoftAttention(tf.keras.layers.Layer):
            def __init__(self,ch,m,concat_with_x=False,aggregate=False,**kwargs):
                self.channels=int(ch)
                self.multiheads = m
                self.aggregate_channels = aggregate
                self.concat_input_with_scaled = concat_with_x

                
                super(SoftAttention,self).__init__(**kwargs)

            def build(self,input_shape):

                self.i_shape = input_shape

                kernel_shape_conv3d = (self.channels, 3, 3) + (1, self.multiheads) # DHWC
            
                self.out_attention_maps_shape = input_shape[0:1]+(self.multiheads,)+input_shape[1:-1]
                
                if self.aggregate_channels==False:

                    self.out_features_shape = input_shape[:-1]+(input_shape[-1]+(input_shape[-1]*self.multiheads),)
                else:
                    if self.concat_input_with_scaled:
                        self.out_features_shape = input_shape[:-1]+(input_shape[-1]*2,)
                    else:
                        self.out_features_shape = input_shape
                

                self.kernel_conv3d = self.add_weight(shape=kernel_shape_conv3d,
                                                initializer='he_uniform',
                                                name='kernel_conv3d')
                self.bias_conv3d = self.add_weight(shape=(self.multiheads,),
                                            initializer='zeros',
                                            name='bias_conv3d')

                super(SoftAttention, self).build(input_shape)

            def call(self, x):

                exp_x = tf.keras.backend.expand_dims(x,axis=-1)

                c3d = tf.keras.backend.conv3d(exp_x,
                            kernel=self.kernel_conv3d,
                            strides=(1,1,self.i_shape[-1]), padding='same', data_format='channels_last')
                conv3d = tf.keras.backend.bias_add(c3d,
                                self.bias_conv3d)
                conv3d = tf.keras.layers.Activation('relu')(conv3d)

                conv3d = tf.keras.backend.permute_dimensions(conv3d,pattern=(0,4,1,2,3))

                
                conv3d = tf.keras.backend.squeeze(conv3d, axis=-1)
                conv3d = tf.keras.backend.reshape(conv3d,shape=(-1, self.multiheads ,self.i_shape[1]*self.i_shape[2]))

                softmax_alpha = tf.keras.backend.softmax(conv3d, axis=-1) 
                softmax_alpha = tf.keras.layers.Reshape(target_shape=(self.multiheads, self.i_shape[1],self.i_shape[2]))(softmax_alpha)

                if self.aggregate_channels==False:
                    exp_softmax_alpha = tf.keras.backend.expand_dims(softmax_alpha, axis=-1)       
                    exp_softmax_alpha = tf.keras.backend.permute_dimensions(exp_softmax_alpha,pattern=(0,2,3,1,4))
        
                    x_exp = tf.keras.backend.expand_dims(x,axis=-2)
        
                    u = tf.keras.layers.Multiply()([exp_softmax_alpha, x_exp])   
        
                    u = tf.keras.layers.Reshape(target_shape=(self.i_shape[1],self.i_shape[2],u.shape[-1]*u.shape[-2]))(u)

                else:
                    exp_softmax_alpha = tf.keras.backend.permute_dimensions(softmax_alpha,pattern=(0,2,3,1))

                    exp_softmax_alpha = tf.keras.backend.sum(exp_softmax_alpha,axis=-1)

                    exp_softmax_alpha = tf.keras.backend.expand_dims(exp_softmax_alpha, axis=-1)

                    u = tf.keras.layers.Multiply()([exp_softmax_alpha, x])   

                if self.concat_input_with_scaled:
                    o = tf.keras.layers.Concatenate(axis=-1)([u,x])
                else:
                    o = u
                
                return [o, softmax_alpha]

            def compute_output_shape(self, input_shape): 
                return [self.out_features_shape, self.out_attention_maps_shape]

            
            def get_config(self):
                return super(SoftAttention,self).get_config()

        def build_VIT(new_img_size=72,patch_size=6,projection_dim=64,num_heads=4,transformer_layers=8,mlp_head_units=[2048,1024]):
            num_patches = (new_img_size // patch_size) ** 2
            transformer_units = [projection_dim * 2,projection_dim]

            def mlp(x, hidden_units, dropout_rate):
                for units in hidden_units:
                    x = tf.keras.layers.Dense(units, activation=tf.keras.activations.gelu)(x)
                    x = tf.keras.layers.Dropout(dropout_rate)(x)
                return x

            class Patches(keras.layers.Layer):
                def __init__(self, patch_size):
                    super().__init__()
                    self.patch_size = patch_size
                def call(self, images):
                    input_shape = tf.shape(images)
                    batch_size = input_shape[0]
                    height = input_shape[1]
                    width = input_shape[2]
                    channels = input_shape[3]
                    num_patches_h = height // self.patch_size
                    num_patches_w = width // self.patch_size
                    patches = tf.image.extract_patches(images=images,sizes=[1,self.patch_size,self.patch_size,1],strides=[1,self.patch_size,self.patch_size,1],rates=[1,1,1,1],padding="VALID")
                    #patches = keras.ops.image.extract_patches(images, size=self.patch_size)
                    patches = tf.reshape(patches,(batch_size,num_patches_h * num_patches_w, self.patch_size * self.patch_size * channels))
                    #patches = keras.ops.reshape(patches,(
                    #    batch_size,
                    #    num_patches_h * num_patches_w,
                    #    self.patch_size * self.patch_size * channels,
                    #))
                    return patches
                def get_config(self):
                    config = super().get_config()
                    config.update({"patch_size": self.patch_size})
                    return config
            class PatchEncoder(keras.layers.Layer):
                def __init__(self, num_patches, projection_dim):
                    super().__init__()
                    self.num_patches = num_patches
                    self.projection = tf.keras.layers.Dense(units=projection_dim)
                    self.position_embedding = tf.keras.layers.Embedding(input_dim=num_patches,output_dim=projection_dim)
                def call(self, patch):
                    positions = tf.expand_dims(tf.range(0,self.num_patches),axis=0)
                    #positions = keras.ops.expand_dims(keras.ops.arange(start=0,stop=self.num_patches,step=1),axis=0)
                    projected_patches = self.projection(patch)
                    encoded = projected_patches + self.position_embedding(positions)
                    return encoded
                def get_config(self):
                    config = super().get_config()
                    config.update({"num_patches": self.num_patches})
                    return config

            data_aug = tf.keras.Sequential(
                [
                    #tf.keras.layers.Normalization(),
                    tf.keras.layers.Resizing(new_img_size,new_img_size),
                    tf.keras.layers.RandomFlip("horizontal"),
                    tf.keras.layers.RandomRotation(factor=0.02),
                    tf.keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2),
                ], name="data_augmentation")

            inputs = tf.keras.Input(shape=self.img_shape)
            augmented = data_aug(inputs)
            patches = Patches(patch_size)(augmented)
            encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
            for _ in range(transformer_layers):
                x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
                attention_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads,key_dim=projection_dim, dropout=0.1)(x1,x1)
                x2 = tf.keras.layers.Add()([attention_output,encoded_patches])
                x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
                x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
                encoded_patches = tf.keras.layers.Add()([x3,x2])
            representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            representation = tf.keras.layers.Flatten()(representation)
            representation = tf.keras.layers.Dropout(0.5)(representation)
            features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
            logits = tf.keras.layers.Dense(self.num_classes,activation="softmax")(features)
            model = tf.keras.Model(inputs=inputs, outputs=logits)
            return model

        print('INIT: Model: ',self.config['model_name'])
        if self.config['model_init_type'] == 'RandNorm':
            initialiser = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=self.config['model_init_seed'])
        elif self.config['model_init_type'] == 'RandUnif':
            initialiser = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=self.config['model_init_seed'])
        elif self.config['model_init_type'] == 'GlorotNorm':
            initialiser = tf.keras.initializers.GlorotNormal(seed=self.config['model_init_seed'])
        elif self.config['model_init_type'] == 'GlorotUnif':
            initialiser = tf.keras.initializers.GlorotUniform(seed=self.config['model_init_seed'])
        elif self.config['model_init_type'] == 'HeNorm':
            initialiser = tf.keras.initializers.HeNormal(seed=self.config['model_init_seed'])
        elif self.config['model_init_type'] == 'HeUnif':
            initialiser = tf.keras.initializers.HeUniform(seed=self.config['model_init_seed'])
        else:
            initialiser = None
            print('Model init type not recognised')

        
        #define the model
        if self.config['model_name'] == "CNN":
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape, kernel_initializer=initialiser),
                tf.keras.layers.MaxPool2D(),
                tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.MaxPool2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Dense(self.num_classes, kernel_initializer=initialiser),
                tf.keras.layers.Softmax()
            ])
            self.output_is_logits = False
            self.new_img_size = self.img_shape
        elif self.config['model_name'] == "CNN4":
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape, kernel_initializer=initialiser),
                tf.keras.layers.MaxPool2D(),
                tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.MaxPool2D(),
                tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.MaxPool2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Dense(self.num_classes,activation='softmax', kernel_initializer=initialiser)
            ])
            self.output_is_logits = False
            self.new_img_size = self.img_shape
        elif self.config['model_name'] == "CNN5":
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape, kernel_initializer=initialiser),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Conv2D(128,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Dense(self.num_classes, kernel_initializer=initialiser),
                tf.keras.layers.Softmax()
            ])
            self.output_is_logits = False
            self.new_img_size = self.img_shape
        elif self.config['model_name'] == "CNN5_Scale-1to1":
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape, kernel_initializer=initialiser),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Conv2D(128,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Dense(self.num_classes,activation='softmax', kernel_initializer=initialiser)
            ])
            self.output_is_logits = False
            self.new_img_size = self.img_shape
        elif self.config['model_name'] == "CNN5_NoPool":
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape, kernel_initializer=initialiser),
                tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Conv2D(128,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Dense(self.num_classes,activation='softmax', kernel_initializer=initialiser)
            ])
            self.output_is_logits = False
            self.new_img_size = self.img_shape
        elif self.config['model_name'] == "CNN5_Dense1":
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape, kernel_initializer=initialiser),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Conv2D(128,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Dense(self.num_classes,activation='softmax', kernel_initializer=initialiser)
            ])
            self.output_is_logits = False
            self.new_img_size = self.img_shape
        elif self.config['model_name'] == "CNN5_Dense2":
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape, kernel_initializer=initialiser),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Conv2D(128,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Dense(self.num_classes,activation='softmax', kernel_initializer=initialiser)
            ])
            self.output_is_logits = False
            self.new_img_size = self.img_shape
        elif self.config['model_name'] == "CNN5_Dense3":
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape, kernel_initializer=initialiser),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Conv2D(128,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Dense(self.num_classes,activation='softmax', kernel_initializer=initialiser)
            ])
            self.output_is_logits = False
            self.new_img_size = self.img_shape
        elif self.config['model_name'] == "CNN5_DenseL":
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape, kernel_initializer=initialiser),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Conv2D(128,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Dense(self.num_classes,activation='softmax', kernel_initializer=initialiser)
            ])
            self.output_is_logits = False
            self.new_img_size = self.img_shape
        elif self.config['model_name'] == "CNN5_DenseXL":
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape, kernel_initializer=initialiser),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Conv2D(128,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Dense(self.num_classes,activation='softmax', kernel_initializer=initialiser)
            ])
            self.output_is_logits = False
            self.new_img_size = self.img_shape
        elif self.config['model_name'] == "CNN5_DenseXXL":
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape, kernel_initializer=initialiser),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Conv2D(128,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1024,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Dense(self.num_classes,activation='softmax', kernel_initializer=initialiser)
            ])
            self.output_is_logits = False
            self.new_img_size = self.img_shape
        elif self.config['model_name'] == "CNN6":
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape, kernel_initializer=initialiser),
                tf.keras.layers.Conv2D(32,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Conv2D(128,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Dense(self.num_classes,activation='softmax', kernel_initializer=initialiser)
            ])
            self.output_is_logits = False
            self.new_img_size = self.img_shape
        elif self.config['model_name'] == "CNN7":
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape, kernel_initializer=initialiser),
                tf.keras.layers.Conv2D(32,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Conv2D(128,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Conv2D(128,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Dense(self.num_classes,activation='softmax', kernel_initializer=initialiser)
            ])
            self.output_is_logits = False
            self.new_img_size = self.img_shape
        elif self.config['model_name'] == "CNN8":
            #8 layer CNN
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape, kernel_initializer=initialiser,padding='same'),
                tf.keras.layers.Conv2D(32,3,activation='relu', kernel_initializer=initialiser,padding='same'),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser,padding='same'),
                tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser,padding='same'),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Conv2D(128,3,activation='relu', kernel_initializer=initialiser,padding='same'),
                tf.keras.layers.Conv2D(128,2,activation='relu', kernel_initializer=initialiser,padding='same'),
                tf.keras.layers.Conv2D(256,3,activation='relu', kernel_initializer=initialiser,padding='same'),
                tf.keras.layers.Conv2D(256,3,activation='relu', kernel_initializer=initialiser,padding='same'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Dense(self.num_classes,activation='softmax', kernel_initializer=initialiser)
            ])
            self.output_is_logits = False
            self.new_img_size = self.img_shape
        elif self.config['model_name'] == "CNN9":
            #8 layer CNN
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape, kernel_initializer=initialiser,padding='same'),
                tf.keras.layers.Conv2D(32,3,activation='relu', kernel_initializer=initialiser,padding='same'),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser,padding='same'),
                tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser,padding='same'),
                tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser,padding='same'),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Conv2D(128,3,activation='relu', kernel_initializer=initialiser,padding='same'),
                tf.keras.layers.Conv2D(128,2,activation='relu', kernel_initializer=initialiser,padding='same'),
                tf.keras.layers.Conv2D(256,3,activation='relu', kernel_initializer=initialiser,padding='same'),
                tf.keras.layers.Conv2D(256,3,activation='relu', kernel_initializer=initialiser,padding='same'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Dense(self.num_classes,activation='softmax', kernel_initializer=initialiser)
            ])
            self.output_is_logits = False
            self.new_img_size = self.img_shape
        elif self.config['model_name'] == "CNN10":
            #8 layer CNN
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape, kernel_initializer=initialiser,padding='same'),
                tf.keras.layers.Conv2D(128,3,activation='relu', kernel_initializer=initialiser,padding='same'),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Conv2D(256,3,activation='relu', kernel_initializer=initialiser,padding='same'),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Dense(self.num_classes,activation='softmax', kernel_initializer=initialiser)
            ])
            self.output_is_logits = False
            self.new_img_size = self.img_shape
        elif self.config['model_name'] == "CNN11":
            #8 layer CNN
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape, kernel_initializer=initialiser,padding='same'),
                tf.keras.layers.Conv2D(256,3,activation='relu', kernel_initializer=initialiser,padding='same'),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Conv2D(256,3,activation='relu', kernel_initializer=initialiser,padding='same'),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Dense(self.num_classes,activation='softmax', kernel_initializer=initialiser)
            ])
            self.output_is_logits = False
            self.new_img_size = self.img_shape
        elif self.config['model_name'] == "CNN11_NoPool":
            #8 layer CNN
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape, kernel_initializer=initialiser,padding='same'),
                tf.keras.layers.Conv2D(256,3,activation='relu', kernel_initializer=initialiser,padding='same'),
                tf.keras.layers.Conv2D(256,3,activation='relu', kernel_initializer=initialiser,padding='same'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Dense(self.num_classes,activation='softmax', kernel_initializer=initialiser)
            ])
            self.output_is_logits = False
            self.new_img_size = self.img_shape
        elif self.config['model_name'] == "CNN12":
            #8 layer CNN
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape, kernel_initializer=initialiser,padding='same'),
                tf.keras.layers.Conv2D(512,3,activation='relu', kernel_initializer=initialiser,padding='same'),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Conv2D(256,3,activation='relu', kernel_initializer=initialiser,padding='same'),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Dense(self.num_classes,activation='softmax', kernel_initializer=initialiser)
            ])
            self.output_is_logits = False
            self.new_img_size = self.img_shape
            
        elif self.config['model_name'] == "ResNet18":
            #build resnet18 model
            inputs = keras.Input(shape=self.img_shape)
            outputs = build_resnet(inputs,[2,2,2,2],self.num_classes,self.config.weight_decay)
            self.model = keras.Model(inputs, outputs)
            self.output_is_logits = False
        elif self.config['model_name'] == "ResNetV1-14":
            #https://www.kaggle.com/code/filippokevin/cifar-10-resnet-14/notebook
            inputs = keras.Input(shape=self.img_shape)
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
            dense2 = tf.keras.layers.Dense(self.num_classes,activation='softmax')(dense1)#maxp
            self.model = tf.keras.models.Model(inputs=inputs,outputs=dense2)
            self.output_is_logits = False
        elif self.config['model_name'] == "TFCNN":
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=self.img_shape),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64,activation='relu'),
                tf.keras.layers.Dense(self.num_classes,activation='softmax')
            ])
            self.output_is_logits = False
        elif self.config['model_name'] == "ACLCNN":
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32,(3,3),activation='elu',input_shape=self.img_shape,padding='same'),
                tf.keras.layers.Conv2D(32,(3,3),activation='elu', padding='same'),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Conv2D(64,(3,3),activation='elu', padding='same'),
                tf.keras.layers.Conv2D(64,(3,3),activation='elu', padding='same'),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Conv2D(128,(3,3),activation='elu', padding='same'),
                tf.keras.layers.Conv2D(128,(3,3),activation='elu', padding='same'),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Conv2D(256,(2,2),activation='elu', padding='same'),
                tf.keras.layers.Conv2D(256,(2,2),activation='elu', padding='same'),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512,activation='elu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(self.num_classes,activation='softmax')])
            self.output_is_logits = False
        elif self.config['model_name'] == "IRv2":
            
            irv2 = tf.keras.applications.InceptionResNetV2(
                include_top=True,
                weights=None,
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classifier_activation="softmax",
            )

            # Excluding the last 28 layers of the model. and using soft attention
            conv = irv2.layers[-28].output
            attention_layer,map2 = SoftAttention(aggregate=True,m=16,concat_with_x=False,ch=int(conv.shape[-1]),name='soft_attention')(conv)
            attention_layer=(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding="same")(attention_layer))
            conv=(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding="same")(conv))

            conv = tf.keras.layers.concatenate([conv,attention_layer])
            conv  = tf.keras.layers.Activation('relu')(conv)
            conv = tf.keras.layers.Dropout(0.5)(conv)

            output = tf.keras.layers.Flatten()(conv)
            output = tf.keras.layers.Dense(self.num_classes, activation='softmax')(output)
            self.model = tf.keras.models.Model(inputs=irv2.input, outputs=output)
            self.output_is_logits = False
        elif self.config['model_name'] == "IRv2_pre":

            irv2 = tf.keras.applications.InceptionResNetV2(
                include_top=True,
                weights="imagenet",
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classifier_activation="softmax",
            )

            # Excluding the last 28 layers of the model. and using soft attention
            conv = irv2.layers[-28].output
            attention_layer,map2 = SoftAttention(aggregate=True,m=16,concat_with_x=False,ch=int(conv.shape[-1]),name='soft_attention')(conv)
            attention_layer=(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding="same")(attention_layer))
            conv=(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding="same")(conv))

            conv = tf.keras.layers.concatenate([conv,attention_layer])
            conv  = tf.keras.layers.Activation('relu')(conv)
            conv = tf.keras.layers.Dropout(0.5)(conv)

            output = tf.keras.layers.Flatten()(conv)
            output = tf.keras.layers.Dense(self.num_classes, activation='softmax')(output)
            self.model = tf.keras.models.Model(inputs=irv2.input, outputs=output)
            self.output_is_logits = False
        elif self.config['model_name'] == "VIT":
            self.model = build_VIT()
            self.output_is_logits = False
        elif self.config['model_name'] == "VGG16":
            self.model = tf.keras.applications.VGG16(
                include_top = True,
                weights = None,
                input_shape = (244,244,3),
                classes = self.num_classes,
                classifier_activation = 'softmax'
            )
            self.output_is_logits = False
            def vgg16_preprocess_input(x):
                #scale to 244x244
                x = tf.image.resize(x,[244,244])
                return tf.keras.applications.vgg16.preprocess_input(x)
            self.pre_process_func = vgg16_preprocess_input
        elif self.config['model_name'] == "VGG19":
            self.model = tf.keras.applications.VGG19(
                include_top = True,
                weights = None,
                input_shape = (244,244,3),
                classes = self.num_classes,
                classifier_activation = 'softmax'
            )
            self.output_is_logits = False
            def vgg19_preprocess_input(x):
                #scale to 244x244
                x = tf.image.resize(x,[244,244])
                return tf.keras.applications.vgg19.preprocess_input(x)
            self.pre_process_func = vgg19_preprocess_input
        elif self.config['model_name'] == "ResNet50":
            self.model = tf.keras.applications.ResNet50(
                include_top = True,
                weights = None,
                input_shape = (244,244,3),
                classes = self.num_classes,
                classifier_activation = 'softmax'
            )
            self.output_is_logits = False
            def resnet_preprocess_input(x):
                #scale to 244x244
                x = tf.image.resize(x,[244,244])
                return tf.keras.applications.resnet.preprocess_input(x)
            self.pre_process_func = resnet_preprocess_input
        elif self.config['model_name'] == "ResNet101":
            self.model = tf.keras.applications.ResNet101(
                include_top = True,
                weights = None,
                input_shape = (244,244,3),
                classes = self.num_classes,
                classifier_activation = 'softmax'
            )
            self.output_is_logits = False
            def resnet_preprocess_input(x):
                #scale to 244x244
                x = tf.image.resize(x,[244,244])
                return tf.keras.applications.resnet.preprocess_input(x)
            self.pre_process_func = resnet_preprocess_input
        elif self.config['model_name'] == "ResNet152":
            self.model = tf.keras.applications.ResNet152(
                include_top = True,
                weights = None,
                input_shape = (244,244,3),
                classes = self.num_classes,
                classifier_activation = 'softmax'
            )
            self.output_is_logits = False
            def resnet_preprocess_input(x):
                #scale to 244x244
                x = tf.image.resize(x,[244,244])
                return tf.keras.applications.resnet.preprocess_input(x)
            self.pre_process_func = resnet_preprocess_input
        elif self.config['model_name'] == "InceptionV3":
            self.model = tf.keras.applications.InceptionV3(
                include_top = True,
                weights = None,
                input_shape = (299,299,3),
                classes = self.num_classes,
                classifier_activation = 'softmax'
            )
            self.output_is_logits = False
            def inception_preprocess_input(x):
                #scale to 299x299
                x = tf.image.resize(x,[299,299])
                return tf.keras.applications.inception_v3.preprocess_input(x)
            self.pre_process_func = inception_preprocess_input
        elif self.config['model_name'] == "MobileNetV2":
            self.model = tf.keras.applications.MobileNetV2(
                include_top = True,
                weights = None,
                alpha = 1.0,
                input_shape = None,
                classes = self.num_classes,
                classifier_activation = 'softmax'
            )
            self.output_is_logits = False
            def mobilenet_preprocess_input(x):
                #scale to 224x224
                x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
                return tf.image.resize(x,[224,224])
            self.pre_process_func = mobilenet_preprocess_input
            self.new_img_size = (224,224,3)
        elif self.config['model_name'] == "EfficientNetB0":
            self.model = tf.keras.applications.EfficientNetB0(
                include_top = True,
                weights = None,
                input_shape = (32,32,3),
                classes = self.num_classes,
                classifier_activation = 'softmax'
            )
            self.output_is_logits = False
            def efficientnet_preprocess_input(x):
                #scale to 32x32
                x = tf.image.resize(x,[32,32])
                return tf.keras.applications.efficientnet.preprocess_input(x)
            self.pre_process_func = efficientnet_preprocess_input
        elif self.config['model_name'] == "EfficientNetB1":

            self.model = tf.keras.applications.EfficientNetB1(
                include_top = True,
                weights = None,
                input_shape = (32,32,3),
                classes = self.num_classes,
                classifier_activation = 'softmax'
            )
            self.output_is_logits = False
            def efficientnet_preprocess_input(x):
                #scale to 32x32
                x = tf.image.resize(x,[32,32])
                return tf.keras.applications.efficientnet.preprocess_input(x)
            self.pre_process_func = efficientnet_preprocess_input

        elif self.config['model_name'] == "Dense1":
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape, kernel_initializer=initialiser),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Conv2D(32,3,activation='relu', kernel_initializer=initialiser),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(self.num_classes, kernel_initializer=initialiser),
                tf.keras.layers.Activation('softmax')
            ])
            self.output_is_logits = False
            self.new_img_size = self.img_shape
        elif self.config['model_name'] == "imdbConv1D":
            #var = [max_features,sequence_length,embedding_dim]
            self.model = tf.keras.Sequential([
                layers.Embedding(vars[0] + 1, vars[2], input_length=vars[1]), 
                layers.Conv1D(128, 5, activation='leaky_relu'),
                layers.MaxPooling1D(2),
                layers.Conv1D(64, 5, activation='leaky_relu'),
                layers.Dropout(0.2),
                layers.GlobalMaxPooling1D(),
                layers.Dense(64, activation='leaky_relu'),
                layers.Dense(1)
            ])
            self.output_is_logits = True
        elif self.config['model_name'] == 'newswireConv1D':
            #var = [max_features,sequence_length,embedding_dim]
            self.model = tf.keras.Sequential([
                layers.Embedding(vars[0] + 1, vars[2], input_length=vars[1]), 
                layers.Conv1D(128, 5, activation='leaky_relu'),
                layers.MaxPooling1D(2),
                layers.Conv1D(64, 5, activation='leaky_relu'),
                layers.Dropout(0.2),
                layers.GlobalMaxPooling1D(),
                layers.Dense(64, activation='leaky_relu'),
                layers.Dense(46, activation='softmax')
            ])
            self.output_is_logits = False

        elif self.config['model_name'] == "speechcommandsCNN":
            self.model = tf.keras.Sequential([
                layers.Input(shape=(124,129,1)),
                layers.Resizing(32,32),
                self.data.norm_layer,
                layers.Conv2D(32,3,activation='relu'),
                layers.Conv2D(64,3,activation='relu'),
                layers.MaxPooling2D(),
                layers.Dropout(0.25),
                layers.Flatten(),
                layers.Dense(128,activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes,activation='softmax')
            ])
            self.output_is_logits = False
        else:
            print('Model not recognised')
        
        if self.config['model_name'] not in ["imdbConv1D","newswireConv1D","speechcommandsCNN"]:
            print('Model built with shape:',self.new_img_size+(1,))
            self.model.build(input_shape=self.new_img_size + (1,))
    
    def count_params(self):
        trainable_params = np.sum([np.prod(v.get_shape()) for v in self.model.trainable_weights])
        #non_trainable_params = np.sum([np.prod(v.get_shape()) for v in self.model.non_trainable_weights])
        #total_params = trainable_params + non_trainable_params
        return trainable_params

    class CustomModel(tf.keras.Model):
        def __init__(self,model,optimizer,loss_func):
            super().__init__()
            self.model = model
            self.optimizer = optimizer
            self.loss_func = loss_func
            self.train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
            self.train_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='train_acc')
            self.train_prec_metric = tf.keras.metrics.Precision(name='train_prec')
            self.train_rec_metric = tf.keras.metrics.Recall(name='train_rec')

            self.test_loss_metric = tf.keras.metrics.Mean(name='test_loss')
            self.test_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='test_acc')
            self.test_prec_metric = tf.keras.metrics.Precision(name='test_prec')
            self.test_rec_metric = tf.keras.metrics.Recall(name='test_rec')
        
        def call(self,inputs,training=False):
            return self.model(inputs,training=training)
        
        def train_step(self,items):
            x,y = items
            loss,y_hat = self.optimizer.train_step(self,x,y)
            
            #update metrics
            self.train_loss_metric.update_state(loss)
            self.train_acc_metric.update_state(y,y_hat)
            self.train_prec_metric.update_state(y,y_hat)
            self.train_rec_metric.update_state(y,y_hat)


        @property
        def metrics(self):
            return [self.train_loss_metric,
                self.train_acc_metric,
                self.train_prec_metric,
                self.train_rec_metric,
                self.test_loss_metric,
                self.test_acc_metric,
                self.test_prec_metric,
                self.test_rec_metric]
            
            


    def model_compile(self):
        #self.model.summary()
        #self.model.compile(optimizer=self.optimizer,loss=self.loss_func)
        self.model = self.CustomModel(self.model,self.optimizer,self.loss_func)
        self.model.compile(optimizer=self.optimizer,loss=self.loss_func,metrics=self.metrics_init())
        print('Model compiled')
        print(self.model.summary())
        #wandb.log({'Model':self.count_params()},step=0)
        

    def lr_schedule(self,epoch,init=False):
        #this needs to define the learning rate schedule [Currently not impoelemented]
        #THIS IS WRONG NEED TO VHANGE LR_DECAY_PARAMS NAME
        if self.config['lr_decay_type'] == 'exp':
            self.lr = tf.keras.optimizers.schedules.ExponentialDecay(self.config['lr'],decay_steps=self.config['lr_decay_type'][0],decay_rate=self.config['lr_decay_type'][1],staircase=True)
        elif self.config['lr_decay_type'] == 'fixed':
            self.lr = self.config['lr']
        elif self.config['lr_decay_type'] == 'cosine':
            self.lr = tf.keras.experimental.CosineDecay(self.config['lr'],self.config['lr_decay_type'][0])
        elif self.config['lr_decay_type'] == 'cosine_restarts':
            self.lr = tf.keras.experimental.CosineDecayRestarts(self.config['lr'],self.config['lr_decay_type'][0])
        else:
            print('Learning rate decay type not recognised')

    
    def epoch_init(self):
        #this is called at the start of each epoch
        #Reset the metrics at the start of the next epoch
        self.train_loss_metric.reset_states()
        self.train_acc_metric.reset_states()
        self.train_prec_metric.reset_states()
        self.train_rec_metric.reset_states()

        self.test_loss_metric.reset_states()
        self.test_acc_metric.reset_states()
        self.test_prec_metric.reset_states()
        self.test_rec_metric.reset_states()
    
    def early_stop(self,adjusted_epoch):
        #this needs to define the early stop
        #returns true if early stop is triggered
        #check test accuracy
        if self.test_acc_metric.result() > self.max_acc:
            self.max_acc = self.test_acc_metric.result()
        
        if adjusted_epoch > self.config.early_stop_epoch:
            if self.test_acc_metric.result() < self.max_acc:
                self.early_stop_count += 1
            else:
                self.early_stop_count = 0

        if self.early_stop_count >= self.config.early_stop:
            print('Early stop triggered')
            return True
        else:
            return False
        
    @tf.function
    def distributed_FIM_step(self,items):
        replica_grads = self.strategy.run(self.Get_Z,args=(items,)) #this should return a list of grads for each replica
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, replica_grads,axis=None) #sum the grads over all replicas

    @tf.function
    def Get_Z(self,items):
        imgs,labels = items
        bs = tf.shape(imgs)[0]
        with tf.GradientTape() as tape:
            y_hat = self.model(imgs,training=False) #get model output (usually softmax) [BS x num_classes]
            selected = tf.squeeze(tf.random.categorical(tf.math.log(y_hat), 1)) #sample from the output [BS x 1]
            output = tf.gather(y_hat,selected,axis=1,batch_dims=1) #get the output for the selected class [BS x 1]
            output = tf.math.log(output) #log the output [BS x 1]
        j = tape.jacobian(output,self.model.trainable_variables) #get the jacobian of the output wrt the model params [BS x num_layer_params x layers]
        layer_sizes = [tf.reduce_sum(tf.size(v)) for v in self.model.trainable_variables] #get the size of each layer
        j = [tf.reshape(j[i],(bs,layer_sizes[i])) for i in range(len(j))] #reshape the jacobian to [BS x num_layer_params x layers]
        j = tf.concat(j,axis=1) #concat the jacobian over the layers [BS x num_params]
        j = tf.square(j) #square the jacobian [BS x num_params]
        j = tf.reduce_sum(j) #sum the jacobian [BS x 1]
        return j 
    

    @tf.function
    def Get_Z_logit(self,items):
        x,y = items
        bs = tf.shape(x)[0]
        with tf.GradientTape() as tape:
            #print(text_batch[i])
            #item = tf.expand_dims(x[i],0)
            y_hat = self.model(x,training=False) #[BS x logits]
            y_hat = tf.nn.sigmoid(y_hat)    #convert to probabilities [BS x probs]
            y_hat = tf.concat([1-y_hat,y_hat],axis=1) #[BS x [0.3,0.7]]  #convert to categorical
            selected = tf.squeeze(tf.random.categorical(tf.math.log(y_hat), 1)) #sample from the output [BS x 1]
            output = tf.gather(y_hat,selected,axis=1,batch_dims=1) #[Bs x 1]
            output = tf.math.log(output)
        
        g = tape.jacobian(output,self.model.trainable_variables)
        layer_sizes = [tf.reduce_sum(tf.size(v)) for v in self.model.trainable_variables] #get the size of each layer
        g = [tf.reshape(g[i],(bs,layer_sizes[i])) for i in range(len(g))] #reshape the gradient to [BS x num_layer_params x layers]
        g = tf.concat(g,axis=1) #concat the gradient over the layers [BS x num_params]
        g = tf.square(g) #square the gradient [BS x num_params]
        g = tf.reduce_sum(g) #sum the gradient [ 1]
        return g

    @tf.function
    def Get_Z_softmax(self,items):
        x,y = items
        bs = tf.shape(x)[0]
        with tf.GradientTape() as tape:
            #print(text_batch[i])
            y_hat = self.model(x,training=False) #get model output (softmax) [BS x num_classes]
            print(y_hat)
            selected = tf.squeeze(tf.random.categorical(tf.math.log(y_hat), 1)) #sample from the output [BS x 1]
            output = tf.gather(y_hat,selected,axis=1,batch_dims=1) #get the output for the selected class [BS x 1]
            output = tf.math.log(output) #log the output [BS x 1]
        
        g = tape.jacobian(output,self.model.trainable_variables)
        layer_sizes = [tf.reduce_sum(tf.size(v)) for v in self.model.trainable_variables] #get the size of each layer
        g = [tf.reshape(g[i],(bs,layer_sizes[i])) for i in range(len(g))] #reshape the gradient to [BS x num_layer_params x layers]
        g = tf.concat(g,axis=1) #concat the gradient over the layers [BS x num_params]
        g = tf.square(g) #square the gradient [BS x num_params]
        g = tf.reduce_sum(g) #sum the gradient [ 1]
        return g

    @tf.function
    def Get_G(self,items):
        #This is the full first Gauss newton matrix (representing the FIM) for cat cross entropy
        #(s_j {i=j} - s_j s_i) dz/dtheta_i dz/dtheta_j and totals [num_params x num_params]
        #S x dz x dz
        #model should not have softmax
        #batch size should be 1 (for now)

        imgs,labels = items
        bs = tf.shape(imgs)[0]
        with tf.GradientTape() as tape1:
            z = tf.squeeze(self.model_nosm(imgs,training=False)) #get the output [num_classes]
            s = tf.nn.softmax(z) #get the softmax output [num_classes]
        num_classes = tf.shape(z)[0]

    
        #S
        S = s[:,None]*s[None,:]
        S = tf.linalg.set_diag(S,s*(1-s)) #[num_classes x num_classes]
        
        #dz/dtheta [Might be able to do this faster with tf.einsum]
        dzdt = tape1.jacobian(z,self.model_nosm.trainable_variables) #[layers x (classes x sublayerparams)]
        dzdt = [tf.reshape(l,[num_classes,-1]) for l in dzdt] #[layers x (classes x layerparams)]
        dzdt = tf.concat(dzdt,axis=1) #[classes x params]

        #S x dz/dtheta
        #G = tf.tensordot(tf.transpose(dzdt),tf.transpose(S),axes=1) #dz/dtheta^T dot S^T [num_params x classes].[classes x num_classes] = [num_params x num_classes]
        #dz/dtheta x S x dz/dtheta (This is large so directly calc trace) (tr())
        #G = tf.tensordot(G,dzdt,axes=1) #multiply G x dz/dtheta = [num_params x num_params]
        #trG_0 = tf.einsum('ic,ci->',G,dzdt) #trace of G [1]
        trS = tf.linalg.trace(S)

        #tr(dzdtT x ST x dzdt)
        trG = tf.einsum('id,dc,ci->',tf.transpose(dzdt),tf.transpose(S),dzdt)
        trdzdt2 = tf.einsum('ic,ci->',tf.transpose(dzdt),dzdt)
        del tape1
        return [trG,trS,trdzdt2]

    
    @tf.function
    def Get_R_nolayerred(self,items):
        imgs,labels = items
        with tf.GradientTape(persistent=True) as tape1:
            with tf.GradientTape() as tape:
                z = tf.squeeze(self.model(imgs,training=False))
                s = tf.nn.softmax(z)
            # Compute first derivative
            S = s - labels #get the residual [num_classes]
            params = tf.concat([tf.reshape(v,[-1]) for v in self.model.trainable_variables],axis=0) # [layerparams]
            print('params',params)
            dy_dtheta = tape.jacobian(z, params) # [num_classes x layers x layerparams]
            print('dy_dtheta',dy_dtheta)
            #dy_dtheta = [tf.reshape(l,[labels.shape[1],-1]) for l in dy_dtheta] # [num_classes x layerparams]
            #dy_dtheta = tf.concat(dy_dtheta,axis=1) # [num_classes x layerparams]
            #print('dy_dtheta',dy_dtheta)

        # Compute second derivative and add to trace
        d2 = tape1.jacobian(dy_dtheta, params) # [num_classes x layerparams x layerparams]
        print(d2)
        #d2 = [tf.reshape(l,[labels.shape[1],-1,-1]) for l in d2] 
        #d2 = tf.concat(d2,axis=1) # [num_classes x layerparams x layerparams]

        #l_shapes = [tf.shape(v) for v in self.model.trainable_variables]
        #print(l_shapes)
        #l_shapes = tf.reduce_prod([tf.reduce_prod(v) for v in l_shapes])
        #print(l_shapes)
        #d2 = tf.reshape(d2, [labels.shape[1], l_shapes, l_shapes])
        #d2 = tf.tensordot(S, d2, axes=1) # [layerparams x layerparams]
        d2 = tf.einsum('ac,cii->', S, d2)

        #d2 = tf.linalg.trace(d2)
        del tape1
        return d2

    def Get_R(self,items):
        #This is the second order Gauss newton matrix term for cat cross entropy
        #tr((S-Y) d2z/dtheta2) (Tried to do this without calcing full seccond order hessian)
        #model should not have softmax
        #batch size should be 1 (for now)
        z,S = self.Get_z_and_s(items)
        C = 0
        for l in range(len(self.model_nosm.trainable_variables)-2):
            l_shape = tf.reduce_prod(tf.shape(self.model_nosm.trainable_variables[l]))
            C += self.Get_layer_d2zdt2(l,S,items)
        return C
        
    @tf.function
    def Get_z_and_s(self,items):
        imgs,labels = items
        with tf.GradientTape() as tape:
            z = tf.squeeze(self.model_nosm(imgs,training=False))
            s = tf.nn.softmax(z) #CHANGED THIS TO NOT SQUEEZE
        S = s - labels #get the residual [num_classes]
        return z,S

    @tf.function
    def Get_layer_d2zdt2(self,l,S,items):
        imgs,labels = items
        with tf.GradientTape(persistent=True) as tape1:
            with tf.GradientTape() as tape:
                z = tf.squeeze(self.model_nosm(imgs,training=False))
            # Compute first derivative
            dy_dtheta = tape.jacobian(z, self.model_nosm.trainable_variables[l]) # [num_classes x layerparams]
        l_shape = tf.shape(self.model_nosm.trainable_variables[l])
        l_shape = tf.reduce_prod(l_shape)

        # Compute second derivative and add to trace
        d2 = tape1.jacobian(dy_dtheta, self.model_nosm.trainable_variables[l]) # [num_classes x layerparams x layerparams]
        d2 = tf.reshape(d2, [labels.shape[1], l_shape, l_shape])
        #d2 = tf.tensordot(S, d2, axes=1) # [layerparams x layerparams]
        d2 = tf.einsum('ac,cii->', S, d2)

        #d2 = tf.linalg.trace(d2)
        del tape1
        return d2


    def make_softmax_model(self):
        self.model_nosm = tf.keras.Model(inputs=self.model.inputs,outputs=self.model.layers[-2].output)
        #self.output_is_logits = True
    
    def add_softmax(self):
        self.model = tf.keras.Sequential([self.model,tf.keras.layers.Softmax()])
        self.output_is_logits = False

    @tf.function
    def Get_H(self,items):
        imgs,labels = items
        bs = tf.shape(imgs)[0]
        with tf.GradientTape(persistent=True) as tape2:
            with tf.GradientTape() as tape1:
                y_hat = self.model(imgs,training=False)
            grads = tape1.gradient(y_hat,self.model.trainable_variables) #[(layerparams)x layers] sums across batch

        hessians = [tape2.jacobian(g, v) for g, v in zip(grads, self.model.trainable_variables)]
        
        return hessians

    @tf.function
    def Get_NME(self,items):
        #this is (S-Y)d2w/d2theta
        #model should not have softmax
        imgs,labels = items
        bs = tf.shape(imgs)[0]
        with tf.GradientTape(persistent=True) as tape2:
            with tf.GradientTape() as tape1:
                z = self.model(imgs,training=False)
                s = tf.nn.softmax(z)
                Y = tf.one_hot(tf.argmax(y_hat,1),self.num_classes) #one hot the output [BS x num_classes]
                S = y_hat - Y #get the residual [BS x num_classes]
            grads = tape1.gradient(y_hat,self.model.trainable_variables) #[(layerparams)x layers] sums across batch
            grads = tape1.gradient(y_hat,self.model.trainable_variables)

    @tf.function
    def get_residuals(self,items):
        imgs,labels = items
        with tf.GradientTape() as tape:
            softmax = self.model(imgs,training=False)
            r = softmax - labels # [BS x num_classes]
        return r

    # @tf.function
    # def train_step(self,data_inputs):
    #     imgs,labels = data_inputs
    #     with tf.GradientTape() as tape:
    #         preds = self.model(imgs,training=True)
    #         loss = self.loss_func(labels,preds)
    #     grads = tape.gradient(loss,self.model.trainable_variables)
    #     self.optimizer.apply_gradients(zip(grads,self.model.trainable_variables))
    #     self.train_loss_metric.update_state(loss)
    #     self.train_acc_metric.update_state(labels,preds)
    #     self.train_prec_metric.update_state(labels,preds)
    #     self.train_rec_metric.update_state(labels,preds)
    @tf.function
    def SAMUpdate(self,items,lam):
        imgs,labels = items
        with tf.GradientTape() as tape:
            preds = self.model(imgs,training=True)
            loss = self.loss_func(labels,preds)
        grads = tape.gradient(loss,self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.model.trainable_variables))
        return loss

    def train_epochs(self,data,epochs,current_epoch):
        data.build_train_iter(bs=32,shuffle=True)
        data.build_test_iter(bs=32,shuffle=False)
        for epoch in range(epochs):
            #self.epoch_init()
            l = 0
            for batch in range(data.train_batches):
                items = data.get_batch(ds_type='train')
                print(items)
                l +=self.train_step(items)
            wandb.log({'train_loss':l/data.train_batches},step=current_epoch+epoch)

            l =0
            for batch in range(data.test_batches):
                items = data.get_batch(ds_type='test')
                self.test_step(items)
            wandb.log({'test_loss':l/data.test_batches},step=current_epoch+epoch)
            #self.log_metrics(self.train_loss_metric.result(),self.test_loss_metric.result(),current_epoch+epoch)
            # if self.early_stop(epoch):
            #     break







    def log_metrics(self,train_loss,test_loss,epoch_num,adjusted_epoch):
        wandb.log({'train_loss':train_loss,
                   'train_acc':self.train_acc_metric.result(),
                   'train_prec':self.train_prec_metric.result(),
                   'train_rec':self.train_rec_metric.result(),
                   'test_loss':test_loss,
                   'test_acc':self.test_acc_metric.result(),
                   'max_test_acc':self.max_acc,
                   'test_prec':self.test_prec_metric.result(),
                   'test_rec':self.test_rec_metric.result(),
                   'lr':self.model.optimizer.learning_rate.numpy(),
                   "adjusted_epoch":adjusted_epoch},
                   step=epoch_num)



    @tf.function
    def compute_loss(self,data_inputs):
        imgs,labels = data_inputs
        with tf.GradientTape() as tape:
            preds = self.model(imgs,training=False)
            loss = self.loss_func(labels,preds)
        return loss

    @tf.function
    def get_batch_loss(self,imgs,labels):
        preds = self.model(imgs,training=False)
        loss = self.loss_func_nored(labels,preds)
        return loss

    

    @tf.function
    def train_step(self,data_inputs):
        imgs,labels = data_inputs
        with tf.GradientTape() as tape:
            preds = self.model(imgs,training=True)
            per_example_loss = self.loss_func(labels,preds)
            loss = tf.nn.compute_average_loss(per_example_loss)
        grads = tape.gradient(loss,self.model.trainable_variables,)
        self.optimizer.apply_gradients(zip(grads,self.model.trainable_variables))
        #self.train_loss_metric.update_state(loss)
        #self.train_acc_metric.update_state(labels,preds)
        #self.train_prec_metric.update_state(labels,preds)
        #self.train_rec_metric.update_state(labels,preds)
        return loss
        

    @tf.function
    def test_step(self,items):
        imgs,labels = items
        with tf.GradientTape() as tape:
            preds = self.model(imgs,training=False)
            loss = self.loss_func(labels,preds)
            loss = tf.nn.compute_average_loss(loss)
        #self.test_acc_metric.update_state(labels,preds)
        #self.test_prec_metric.update_state(labels,preds)
        #self.test_rec_metric.update_state(labels,preds)
        #self.test_loss_metric.update_state(loss)
        return  loss
        

    @tf.function
    def distributed_get_loss_step(self,data_inputs):
        per_replica_losses = self.strategy.run(self.compute_loss,args=(data_inputs,))
        return per_replica_losses.values

    @tf.function
    def distributed_train_step(self,data_inputs): #imgsand labels are dist batches
        per_replica_losses = self.strategy.run(self.train_step,args=(data_inputs,)) #run the train step on each replica
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,axis=None)
    
    @tf.function
    def distributed_test_step(self,data_inputs):
        per_replica_losses = self.strategy.run(self.test_step,args=(data_inputs,))#run the train step on each replica
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,axis=None)

    @tf.function
    def norm_train_step(self,imgs,labels):
        with tf.GradientTape() as tape:
            preds = self.model(imgs,training=True)
            loss = self.no_reduction_loss_func(labels,preds) #should return the loss of all items in batch
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
    #this is the same as above?
    def get_batch_loss(self,imgs,labels,training=False):
        preds = self.model(imgs,training=training)
        loss = self.loss_func(labels,preds)
        return loss

