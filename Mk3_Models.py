
#Update from mk2 focus on maintianing higher level functions

#Importing Libraries
import tensorflow as tf
import wandb   
from tensorflow import keras
from keras import layers

import math
import time
import numpy as np

import Mk2_Losses as custom_losses
import Mk2_Data as custom_data



class Model(tf.keras.Model):
    def __init__(self,model,config):
        super().__init__()
        self.model = model
        self.config = config
        self.load_metrics(self.config)

    def load_metrics(self,config):
        self.metrics_list = []
        #loss logs
        self.metrics_list.append(tf.keras.metrics.Mean(name='loss'))
        #accuracy logs
        if config['loss_func'] == 'categorical_crossentropy':
            self.metrics_list.append(tf.keras.metrics.CategoricalAccuracy(name='accuracy'))


    def compile(self,optimizer,loss,metrics=None):
        super().compile(optimizer=optimizer,loss=loss,metrics=metrics)

    @tf.function
    def train_step(self, data):
        x, y = data
        if self.config['optimizer'] == 'SGD':
            with tf.GradientTape() as tape:
                y_hat = self.model(x, training=True)  # Forward pass
                loss = self.compiled_loss(y, y_hat)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        elif self.config['optimizer'] == 'SAM_SGD':
            loss, y_hat = self.optimizer.step(x,y,self.model,self.compiled_loss)
        
        else:
            print('Optimizer not recognised')

        #update metrics
        for metric in self.metrics_list:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_hat)
        return {m.name: m.result() for m in self.metrics}
    
    @tf.function
    def test_step(self, data):
        x, y = data
        y_hat = self.model(x, training=False)
        loss = self.compiled_loss(y, y_hat)
        #self.metrics.update_state(y,y_pred,loss)
        for metric in self.metrics_list:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_hat)
        return {m.name: m.result() for m in self.metrics}
    
    def call(self, inputs, training=False):
        return self.model(inputs, training)
        
    @property
    def metrics(self):
        return self.metrics_list

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
    def Get_Z_logits(self,items):
        imgs,labels = items
        bs = tf.shape(imgs)[0]
        with tf.GradientTape() as tape:
            y_hat = self.model(imgs,training=False) #get model output  [BS x num_classes]
            y_hat = tf.nn.softmax(y_hat) #softmax the output [BS x num_classes]
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
    



def build_model(config):
    selected_model,output_is_logits = model_selector(config['model_name'],config)
    learning_schedule = lr_selector(config['lr_decay_type'],config)
    loss_func = loss_selector(config['loss_func'],config,output_is_logits)
    optimizer = optimizer_selector(config['optimizer'],config,lr_schedule=learning_schedule)
    metrics = metric_selector(config)

    model = Model(selected_model,config)
    model.compile(optimizer=optimizer, loss=loss_func, metrics=[metrics])
    return model

class SAM(tf.keras.optimizers.Optimizer):
    def __init__(self, base_optim, rho=0.05, name="SAM", **kwargs):
        super().__init__(name, **kwargs)
        self.base_optim = base_optim
        self.rho = rho  # ball size
        self.title = f"SAM_SGD"

    @tf.function
    def max_step(self,model,x,y,loss_func):
        #compute grads at current point and move to the maximum in the ball
        with tf.GradientTape() as tape:
            y_hat = model(x,training=True)
            loss = loss_func(y,y_hat)
        gs = tape.gradient(loss, model.trainable_variables)
        grad_norm = tf.linalg.global_norm(gs)
        eps = [(g * self.rho)/ (grad_norm + 1e-12) for g in gs]

        for e, var in zip(eps, model.trainable_variables):
            var.assign_add(e)
        #print('EPS: ',[e.shape for e in self.eps])
        #print('Model Trainable Variables: ',[e.shape for e in model.trainable_variables])

        #model.trainable_variables = tf.map_fn(lambda var,eps: var + eps, (model.trainable_variables, self.eps))
        return loss,y_hat,eps
    
    @tf.function
    def min_step(self,model,x,y,loss_func,eps):
        with tf.GradientTape() as tape:
            y_hat = model(x,training=True)
            loss = loss_func(y,y_hat)
        gs = tape.gradient(loss, model.trainable_variables)
        #move back to the original point
        for e, var in zip(eps, model.trainable_variables):
            var.assign_sub(e)
        #model.trainable_variables = tf.map_fn(lambda var,eps: var - eps, (model.trainable_variables, self.eps))
        #apply normal gradient step
        self.base_optim.apply_gradients(zip(gs, model.trainable_variables))

    def step(self, x, y, model, loss_func):
        #compute the max step
        loss,y_hat,eps = self.max_step(model,x,y,loss_func)
        self.min_step(model,x,y,loss_func,eps)
        return loss,y_hat


def metric_selector(config):
    match config['loss_func']:
        case 'categorical_crossentropy':
            return tf.keras.metrics.CategoricalAccuracy()


def loss_selector(loss_name, config, output_is_logits=False):
    match loss_name:
        case 'categorical_crossentropy':
            return tf.keras.losses.CategoricalCrossentropy(from_logits=output_is_logits)

def lr_selector(lr_name,config):
    match lr_name:
        case 'fixed':
            return config['lr']
        case 'exp_decay':
            return tf.keras.optimizers.schedules.ExponentialDecay(config['lr'],decay_steps=config['lr_decay_type'][0],decay_rate=config['lr_decay_type'][1],staircase=True)

def model_selector(model_name,config):
    #this needs to define the model
    #returns the model and the output_is_logits flag
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

    print('INIT: Model: ',config['model_name'])

    if config['model_init_type'] == 'RandNorm':
        initialiser = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=config['model_init_seed'])
    elif config['model_init_type'] == 'RandUnif':
        initialiser = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=config['model_init_seed'])
    elif config['model_init_type'] == 'GlorotNorm':
        initialiser = tf.keras.initializers.GlorotNormal(seed=config['model_init_seed'])
    elif config['model_init_type'] == 'GlorotUnif':
        initialiser = tf.keras.initializers.GlorotUniform(seed=config['model_init_seed'])
    elif config['model_init_type'] == 'HeNorm':
        initialiser = tf.keras.initializers.HeNormal(seed=config['model_init_seed'])
    elif config['model_init_type'] == 'HeUnif':
        initialiser = tf.keras.initializers.HeUniform(seed=config['model_init_seed'])
    else:
        initialiser = None
        print('Model init type not recognised')

    
    #define the model
    if config['model_name'] == "CNN":
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=config['img_size'], kernel_initializer=initialiser),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
            tf.keras.layers.Dense(config['num_classes'], kernel_initializer=initialiser),
            tf.keras.layers.Softmax()
        ])
        output_is_logits = False

    # elif config['model_name'] == "CNN4":
    #     self.model = tf.keras.Sequential([
    #         tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape, kernel_initializer=initialiser),
    #         tf.keras.layers.MaxPool2D(),
    #         tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.MaxPool2D(),
    #         tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.MaxPool2D(),
    #         tf.keras.layers.Flatten(),
    #         tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Dense(self.num_classes,activation='softmax', kernel_initializer=initialiser)
    #     ])
    #     self.output_is_logits = False
    #     self.new_img_size = self.img_shape
    # elif config['model_name'] == "CNN5":
    #     self.model = tf.keras.Sequential([
    #         tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape, kernel_initializer=initialiser),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Conv2D(128,3,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Flatten(),
    #         tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Dense(self.num_classes, kernel_initializer=initialiser),
    #         tf.keras.layers.Softmax()
    #     ])
    #     self.output_is_logits = False
    #     self.new_img_size = self.img_shape
    # elif config['model_name'] == "CNN5_Scale-1to1":
    #     self.model = tf.keras.Sequential([
    #         tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape, kernel_initializer=initialiser),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Conv2D(128,3,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Flatten(),
    #         tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Dense(self.num_classes,activation='softmax', kernel_initializer=initialiser)
    #     ])
    #     self.output_is_logits = False
    #     self.new_img_size = self.img_shape
    # elif config['model_name'] == "CNN5_NoPool":
    #     self.model = tf.keras.Sequential([
    #         tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape, kernel_initializer=initialiser),
    #         tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Conv2D(128,3,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Flatten(),
    #         tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Dense(self.num_classes,activation='softmax', kernel_initializer=initialiser)
    #     ])
    #     self.output_is_logits = False
    #     self.new_img_size = self.img_shape
    # elif config['model_name'] == "CNN5_Dense1":
    #     self.model = tf.keras.Sequential([
    #         tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape, kernel_initializer=initialiser),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Conv2D(128,3,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Flatten(),
    #         tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Dense(self.num_classes,activation='softmax', kernel_initializer=initialiser)
    #     ])
    #     self.output_is_logits = False
    #     self.new_img_size = self.img_shape
    # elif config['model_name'] == "CNN5_Dense2":
    #     self.model = tf.keras.Sequential([
    #         tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape, kernel_initializer=initialiser),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Conv2D(128,3,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Flatten(),
    #         tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Dense(self.num_classes,activation='softmax', kernel_initializer=initialiser)
    #     ])
    #     self.output_is_logits = False
    #     self.new_img_size = self.img_shape
    # elif config['model_name'] == "CNN5_Dense3":
    #     self.model = tf.keras.Sequential([
    #         tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape, kernel_initializer=initialiser),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Conv2D(128,3,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Flatten(),
    #         tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Dense(self.num_classes,activation='softmax', kernel_initializer=initialiser)
    #     ])
    #     self.output_is_logits = False
    #     self.new_img_size = self.img_shape
    # elif config['model_name'] == "CNN5_DenseL":
    #     self.model = tf.keras.Sequential([
    #         tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape, kernel_initializer=initialiser),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Conv2D(128,3,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Flatten(),
    #         tf.keras.layers.Dense(256,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Dense(self.num_classes,activation='softmax', kernel_initializer=initialiser)
    #     ])
    #     self.output_is_logits = False
    #     self.new_img_size = self.img_shape
    # elif config['model_name'] == "CNN5_DenseXL":
    #     self.model = tf.keras.Sequential([
    #         tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape, kernel_initializer=initialiser),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Conv2D(128,3,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Flatten(),
    #         tf.keras.layers.Dense(512,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Dense(self.num_classes,activation='softmax', kernel_initializer=initialiser)
    #     ])
    #     self.output_is_logits = False
    #     self.new_img_size = self.img_shape
    # elif config['model_name'] == "CNN5_DenseXXL":
    #     self.model = tf.keras.Sequential([
    #         tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape, kernel_initializer=initialiser),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Conv2D(128,3,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Flatten(),
    #         tf.keras.layers.Dense(1024,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Dense(self.num_classes,activation='softmax', kernel_initializer=initialiser)
    #     ])
    #     self.output_is_logits = False
    #     self.new_img_size = self.img_shape
    # elif config['model_name'] == "CNN6":
    #     self.model = tf.keras.Sequential([
    #         tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape, kernel_initializer=initialiser),
    #         tf.keras.layers.Conv2D(32,3,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Conv2D(128,3,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Flatten(),
    #         tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Dense(self.num_classes,activation='softmax', kernel_initializer=initialiser)
    #     ])
    #     self.output_is_logits = False
    #     self.new_img_size = self.img_shape
    # elif config['model_name'] == "CNN7":
    #     self.model = tf.keras.Sequential([
    #         tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape, kernel_initializer=initialiser),
    #         tf.keras.layers.Conv2D(32,3,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Conv2D(128,3,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Conv2D(128,3,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Flatten(),
    #         tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Dense(self.num_classes,activation='softmax', kernel_initializer=initialiser)
    #     ])
    #     self.output_is_logits = False
    #     self.new_img_size = self.img_shape
    # elif config['model_name'] == "CNN8":
    #     #8 layer CNN
    #     self.model = tf.keras.Sequential([
    #         tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape, kernel_initializer=initialiser,padding='same'),
    #         tf.keras.layers.Conv2D(32,3,activation='relu', kernel_initializer=initialiser,padding='same'),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser,padding='same'),
    #         tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser,padding='same'),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Conv2D(128,3,activation='relu', kernel_initializer=initialiser,padding='same'),
    #         tf.keras.layers.Conv2D(128,2,activation='relu', kernel_initializer=initialiser,padding='same'),
    #         tf.keras.layers.Conv2D(256,3,activation='relu', kernel_initializer=initialiser,padding='same'),
    #         tf.keras.layers.Conv2D(256,3,activation='relu', kernel_initializer=initialiser,padding='same'),
    #         tf.keras.layers.Flatten(),
    #         tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Dense(self.num_classes,activation='softmax', kernel_initializer=initialiser)
    #     ])
    #     self.output_is_logits = False
    #     self.new_img_size = self.img_shape
    # elif config['model_name'] == "CNN9":
    #     #8 layer CNN
    #     self.model = tf.keras.Sequential([
    #         tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape, kernel_initializer=initialiser,padding='same'),
    #         tf.keras.layers.Conv2D(32,3,activation='relu', kernel_initializer=initialiser,padding='same'),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser,padding='same'),
    #         tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser,padding='same'),
    #         tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser,padding='same'),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Conv2D(128,3,activation='relu', kernel_initializer=initialiser,padding='same'),
    #         tf.keras.layers.Conv2D(128,2,activation='relu', kernel_initializer=initialiser,padding='same'),
    #         tf.keras.layers.Conv2D(256,3,activation='relu', kernel_initializer=initialiser,padding='same'),
    #         tf.keras.layers.Conv2D(256,3,activation='relu', kernel_initializer=initialiser,padding='same'),
    #         tf.keras.layers.Flatten(),
    #         tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Dense(self.num_classes,activation='softmax', kernel_initializer=initialiser)
    #     ])
    #     self.output_is_logits = False
    #     self.new_img_size = self.img_shape
    # elif config['model_name'] == "CNN10":
    #     #8 layer CNN
    #     self.model = tf.keras.Sequential([
    #         tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape, kernel_initializer=initialiser,padding='same'),
    #         tf.keras.layers.Conv2D(128,3,activation='relu', kernel_initializer=initialiser,padding='same'),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Conv2D(256,3,activation='relu', kernel_initializer=initialiser,padding='same'),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Flatten(),
    #         tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Dense(self.num_classes,activation='softmax', kernel_initializer=initialiser)
    #     ])
    #     self.output_is_logits = False
    #     self.new_img_size = self.img_shape
    # elif config['model_name'] == "CNN11":
    #     #8 layer CNN
    #     self.model = tf.keras.Sequential([
    #         tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape, kernel_initializer=initialiser,padding='same'),
    #         tf.keras.layers.Conv2D(256,3,activation='relu', kernel_initializer=initialiser,padding='same'),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Conv2D(256,3,activation='relu', kernel_initializer=initialiser,padding='same'),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Flatten(),
    #         tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Dense(self.num_classes,activation='softmax', kernel_initializer=initialiser)
    #     ])
    #     self.output_is_logits = False
    #     self.new_img_size = self.img_shape
    # elif config['model_name'] == "CNN11_NoPool":
    #     #8 layer CNN
    #     self.model = tf.keras.Sequential([
    #         tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape, kernel_initializer=initialiser,padding='same'),
    #         tf.keras.layers.Conv2D(256,3,activation='relu', kernel_initializer=initialiser,padding='same'),
    #         tf.keras.layers.Conv2D(256,3,activation='relu', kernel_initializer=initialiser,padding='same'),
    #         tf.keras.layers.Flatten(),
    #         tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Dense(self.num_classes,activation='softmax', kernel_initializer=initialiser)
    #     ])
    #     self.output_is_logits = False
    #     self.new_img_size = self.img_shape
    # elif config['model_name'] == "CNN12":
    #     #8 layer CNN
    #     self.model = tf.keras.Sequential([
    #         tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.img_shape, kernel_initializer=initialiser,padding='same'),
    #         tf.keras.layers.Conv2D(512,3,activation='relu', kernel_initializer=initialiser,padding='same'),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Conv2D(256,3,activation='relu', kernel_initializer=initialiser,padding='same'),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Flatten(),
    #         tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
    #         tf.keras.layers.Dense(self.num_classes,activation='softmax', kernel_initializer=initialiser)
    #     ])
    #     self.output_is_logits = False
    #     self.new_img_size = self.img_shape
        
    # elif config['model_name'] == "ResNet18":
    #     #build resnet18 model
    #     inputs = keras.Input(shape=self.img_shape)
    #     outputs = build_resnet(inputs,[2,2,2,2],self.num_classes,config.weight_decay)
    #     self.model = keras.Model(inputs, outputs)
    #     self.output_is_logits = False
    # elif config['model_name'] == "ResNetV1-14":
    #     #https://www.kaggle.com/code/filippokevin/cifar-10-resnet-14/notebook
    #     inputs = keras.Input(shape=self.img_shape)
    #     conv_1 = tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding="same")(inputs)
    #     conv_b1_1 = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding="same")(conv_1)
    #     conv_b1_2 = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding="same")(conv_b1_1)
    #     conv_b1_3 = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding="same")(conv_b1_2)
    #     sum_1 = tf.keras.layers.Concatenate()([conv_1,conv_b1_3])
    #     avg_1 = tf.keras.layers.AveragePooling2D(pool_size=(2,2))(sum_1)
    #     conv_b2_1 = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding="same")(avg_1)
    #     conv_b2_2 = tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding="same")(conv_b2_1)
    #     conv_b2_3 = tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding="same")(conv_b2_2)
    #     sum_2 = tf.keras.layers.Concatenate()([avg_1,conv_b2_3])
    #     avg_2 = tf.keras.layers.AveragePooling2D(pool_size=(2,2))(sum_2)
    #     conv_b3_1 = tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3),activation='relu',padding="same")(avg_2)
    #     conv_b3_2 = tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3),activation='relu',padding="same")(conv_b3_1)
    #     conv_b3_3 = tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3),activation='relu',padding="same")(conv_b3_2)
    #     sum_3 = tf.keras.layers.Concatenate()([avg_2,conv_b3_3])
    #     avg_3 = tf.keras.layers.AveragePooling2D(pool_size=(2,2))(sum_3)
    #     conv_b4_1 = tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),activation='relu',padding="same")(avg_3)
    #     conv_b4_2 = tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),activation='relu',padding="same")(conv_b4_1)
    #     conv_b4_3 = tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),activation='relu',padding="same")(conv_b4_2)
    #     sum_4 = tf.keras.layers.Concatenate()([avg_3,conv_b4_3])
    #     avg = tf.keras.layers.AveragePooling2D(pool_size=(2,2))(sum_4)
    #     flat = tf.keras.layers.Flatten()(avg)#problema <--
    #     dense1 = tf.keras.layers.Dense(16,activation='relu')(flat)
    #     dense2 = tf.keras.layers.Dense(self.num_classes,activation='softmax')(dense1)#maxp
    #     self.model = tf.keras.models.Model(inputs=inputs,outputs=dense2)
    #     self.output_is_logits = False
    # elif config['model_name'] == "TFCNN":
    #     self.model = tf.keras.Sequential([
    #         tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=self.img_shape),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    #         tf.keras.layers.Flatten(),
    #         tf.keras.layers.Dense(64,activation='relu'),
    #         tf.keras.layers.Dense(self.num_classes,activation='softmax')
    #     ])
    #     self.output_is_logits = False
    # elif config['model_name'] == "ACLCNN":
    #     self.model = tf.keras.Sequential([
    #         tf.keras.layers.Conv2D(32,(3,3),activation='elu',input_shape=self.img_shape,padding='same'),
    #         tf.keras.layers.Conv2D(32,(3,3),activation='elu', padding='same'),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Dropout(0.25),
    #         tf.keras.layers.Conv2D(64,(3,3),activation='elu', padding='same'),
    #         tf.keras.layers.Conv2D(64,(3,3),activation='elu', padding='same'),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Dropout(0.25),
    #         tf.keras.layers.Conv2D(128,(3,3),activation='elu', padding='same'),
    #         tf.keras.layers.Conv2D(128,(3,3),activation='elu', padding='same'),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Dropout(0.25),
    #         tf.keras.layers.Conv2D(256,(2,2),activation='elu', padding='same'),
    #         tf.keras.layers.Conv2D(256,(2,2),activation='elu', padding='same'),
    #         tf.keras.layers.MaxPool2D((2,2)),
    #         tf.keras.layers.Dropout(0.25),
    #         tf.keras.layers.Flatten(),
    #         tf.keras.layers.Dense(512,activation='elu'),
    #         tf.keras.layers.Dropout(0.5),
    #         tf.keras.layers.Dense(self.num_classes,activation='softmax')])
    #     self.output_is_logits = False
    # elif config['model_name'] == "IRv2":
        
    #     irv2 = tf.keras.applications.InceptionResNetV2(
    #         include_top=True,
    #         weights=None,
    #         input_tensor=None,
    #         input_shape=None,
    #         pooling=None,
    #         classifier_activation="softmax",
    #     )

    #     # Excluding the last 28 layers of the model. and using soft attention
    #     conv = irv2.layers[-28].output
    #     attention_layer,map2 = SoftAttention(aggregate=True,m=16,concat_with_x=False,ch=int(conv.shape[-1]),name='soft_attention')(conv)
    #     attention_layer=(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding="same")(attention_layer))
    #     conv=(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding="same")(conv))

    #     conv = tf.keras.layers.concatenate([conv,attention_layer])
    #     conv  = tf.keras.layers.Activation('relu')(conv)
    #     conv = tf.keras.layers.Dropout(0.5)(conv)

    #     output = tf.keras.layers.Flatten()(conv)
    #     output = tf.keras.layers.Dense(self.num_classes, activation='softmax')(output)
    #     self.model = tf.keras.models.Model(inputs=irv2.input, outputs=output)
    #     self.output_is_logits = False
    # elif config['model_name'] == "IRv2_pre":

    #     irv2 = tf.keras.applications.InceptionResNetV2(
    #         include_top=True,
    #         weights="imagenet",
    #         input_tensor=None,
    #         input_shape=None,
    #         pooling=None,
    #         classifier_activation="softmax",
    #     )

    #     # Excluding the last 28 layers of the model. and using soft attention
    #     conv = irv2.layers[-28].output
    #     attention_layer,map2 = SoftAttention(aggregate=True,m=16,concat_with_x=False,ch=int(conv.shape[-1]),name='soft_attention')(conv)
    #     attention_layer=(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding="same")(attention_layer))
    #     conv=(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding="same")(conv))

    #     conv = tf.keras.layers.concatenate([conv,attention_layer])
    #     conv  = tf.keras.layers.Activation('relu')(conv)
    #     conv = tf.keras.layers.Dropout(0.5)(conv)

    #     output = tf.keras.layers.Flatten()(conv)
    #     output = tf.keras.layers.Dense(self.num_classes, activation='softmax')(output)
    #     self.model = tf.keras.models.Model(inputs=irv2.input, outputs=output)
    #     self.output_is_logits = False
    # elif config['model_name'] == "VIT":
    #     self.model = build_VIT()
    #     self.output_is_logits = False
    # elif config['model_name'] == "VGG16":
    #     self.model = tf.keras.applications.VGG16(
    #         include_top = True,
    #         weights = None,
    #         input_shape = (244,244,3),
    #         classes = self.num_classes,
    #         classifier_activation = 'softmax'
    #     )
    #     self.output_is_logits = False
    #     def vgg16_preprocess_input(x):
    #         #scale to 244x244
    #         x = tf.image.resize(x,[244,244])
    #         return tf.keras.applications.vgg16.preprocess_input(x)
    #     self.pre_process_func = vgg16_preprocess_input
    # elif config['model_name'] == "VGG19":
    #     self.model = tf.keras.applications.VGG19(
    #         include_top = True,
    #         weights = None,
    #         input_shape = (244,244,3),
    #         classes = self.num_classes,
    #         classifier_activation = 'softmax'
    #     )
    #     self.output_is_logits = False
    #     def vgg19_preprocess_input(x):
    #         #scale to 244x244
    #         x = tf.image.resize(x,[244,244])
    #         return tf.keras.applications.vgg19.preprocess_input(x)
    #     self.pre_process_func = vgg19_preprocess_input
    # elif config['model_name'] == "ResNet50":
    #     self.model = tf.keras.applications.ResNet50(
    #         include_top = True,
    #         weights = None,
    #         input_shape = (244,244,3),
    #         classes = self.num_classes,
    #         classifier_activation = 'softmax'
    #     )
    #     self.output_is_logits = False
    #     def resnet_preprocess_input(x):
    #         #scale to 244x244
    #         x = tf.image.resize(x,[244,244])
    #         return tf.keras.applications.resnet.preprocess_input(x)
    #     self.pre_process_func = resnet_preprocess_input
    # elif config['model_name'] == "ResNet101":
    #     self.model = tf.keras.applications.ResNet101(
    #         include_top = True,
    #         weights = None,
    #         input_shape = (244,244,3),
    #         classes = self.num_classes,
    #         classifier_activation = 'softmax'
    #     )
    #     self.output_is_logits = False
    #     def resnet_preprocess_input(x):
    #         #scale to 244x244
    #         x = tf.image.resize(x,[244,244])
    #         return tf.keras.applications.resnet.preprocess_input(x)
    #     self.pre_process_func = resnet_preprocess_input
    # elif config['model_name'] == "ResNet152":
    #     self.model = tf.keras.applications.ResNet152(
    #         include_top = True,
    #         weights = None,
    #         input_shape = (244,244,3),
    #         classes = self.num_classes,
    #         classifier_activation = 'softmax'
    #     )
    #     self.output_is_logits = False
    #     def resnet_preprocess_input(x):
    #         #scale to 244x244
    #         x = tf.image.resize(x,[244,244])
    #         return tf.keras.applications.resnet.preprocess_input(x)
    #     self.pre_process_func = resnet_preprocess_input
    # elif config['model_name'] == "InceptionV3":
    #     self.model = tf.keras.applications.InceptionV3(
    #         include_top = True,
    #         weights = None,
    #         input_shape = (299,299,3),
    #         classes = self.num_classes,
    #         classifier_activation = 'softmax'
    #     )
    #     self.output_is_logits = False
    #     def inception_preprocess_input(x):
    #         #scale to 299x299
    #         x = tf.image.resize(x,[299,299])
    #         return tf.keras.applications.inception_v3.preprocess_input(x)
    #     self.pre_process_func = inception_preprocess_input
    # elif config['model_name'] == "MobileNetV2":
    #     self.model = tf.keras.applications.MobileNetV2(
    #         include_top = True,
    #         weights = None,
    #         alpha = 1.0,
    #         input_shape = None,
    #         classes = self.num_classes,
    #         classifier_activation = 'softmax'
    #     )
    #     self.output_is_logits = False
    #     def mobilenet_preprocess_input(x):
    #         #scale to 224x224
    #         x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    #         return tf.image.resize(x,[224,224])
    #     self.pre_process_func = mobilenet_preprocess_input
    #     self.new_img_size = (224,224,3)
    # elif config['model_name'] == "EfficientNetB0":
    #     self.model = tf.keras.applications.EfficientNetB0(
    #         include_top = True,
    #         weights = None,
    #         input_shape = (32,32,3),
    #         classes = self.num_classes,
    #         classifier_activation = 'softmax'
    #     )
    #     self.output_is_logits = False
    #     def efficientnet_preprocess_input(x):
    #         #scale to 32x32
    #         x = tf.image.resize(x,[32,32])
    #         return tf.keras.applications.efficientnet.preprocess_input(x)
    #     self.pre_process_func = efficientnet_preprocess_input
    # elif config['model_name'] == "EfficientNetB1":

    #     self.model = tf.keras.applications.EfficientNetB1(
    #         include_top = True,
    #         weights = None,
    #         input_shape = (32,32,3),
    #         classes = self.num_classes,
    #         classifier_activation = 'softmax'
    #     )
    #     self.output_is_logits = False
    #     def efficientnet_preprocess_input(x):
    #         #scale to 32x32
    #         x = tf.image.resize(x,[32,32])
    #         return tf.keras.applications.efficientnet.preprocess_input(x)
    #     self.pre_process_func = efficientnet_preprocess_input

    elif config['model_name'] == "Dense1":
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=config['img_size'], kernel_initializer=initialiser),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(32,3,activation='relu', kernel_initializer=initialiser),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(config['num_classes'], kernel_initializer=initialiser),
            tf.keras.layers.Activation('softmax')
        ])
        output_is_logits = False
    # elif config['model_name'] == "imdbConv1D":
    #     #var = [max_features,sequence_length,embedding_dim]
    #     self.model = tf.keras.Sequential([
    #         layers.Embedding(vars[0] + 1, vars[2], input_length=vars[1]), 
    #         layers.Conv1D(128, 5, activation='leaky_relu'),
    #         layers.MaxPooling1D(2),
    #         layers.Conv1D(64, 5, activation='leaky_relu'),
    #         layers.Dropout(0.2),
    #         layers.GlobalMaxPooling1D(),
    #         layers.Dense(64, activation='leaky_relu'),
    #         layers.Dense(1)
    #     ])
    #     self.output_is_logits = True
    # elif config['model_name'] == 'newswireConv1D':
    #     #var = [max_features,sequence_length,embedding_dim]
    #     self.model = tf.keras.Sequential([
    #         layers.Embedding(vars[0] + 1, vars[2], input_length=vars[1]), 
    #         layers.Conv1D(128, 5, activation='leaky_relu'),
    #         layers.MaxPooling1D(2),
    #         layers.Conv1D(64, 5, activation='leaky_relu'),
    #         layers.Dropout(0.2),
    #         layers.GlobalMaxPooling1D(),
    #         layers.Dense(64, activation='leaky_relu'),
    #         layers.Dense(46, activation='softmax')
    #     ])
    #     self.output_is_logits = False

    # elif config['model_name'] == "speechcommandsCNN":
        # self.model = tf.keras.Sequential([
        #     layers.Input(shape=(124,129,1)),
        #     layers.Resizing(32,32),
        #     self.data.norm_layer,
        #     layers.Conv2D(32,3,activation='relu'),
        #     layers.Conv2D(64,3,activation='relu'),
        #     layers.MaxPooling2D(),
        #     layers.Dropout(0.25),
        #     layers.Flatten(),
        #     layers.Dense(128,activation='relu'),
        #     layers.Dropout(0.5),
        #     layers.Dense(self.num_classes,activation='softmax')
        # ])
        # self.output_is_logits = False
    else:
        print('Model not recognised')
    
    # if config['model_name'] not in ["imdbConv1D","newswireConv1D","speechcommandsCNN"]:
    #     print('Model built with shape:',self.new_img_size+(1,))
    #     self.model.build(input_shape=self.new_img_size + (1,))
    return model, output_is_logits

def optimizer_selector(optimizer_name,config,lr_schedule):
    match optimizer_name:
        case 'SGD':
            return tf.keras.optimizers.SGD(learning_rate=lr_schedule)
        case 'SAM_SGD':
            return SAM(tf.keras.optimizers.SGD(learning_rate=lr_schedule),rho=config['rho'])