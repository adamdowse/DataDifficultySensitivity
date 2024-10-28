
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
import Mk2_Funcs as custom_funcs



class Model(tf.keras.Model):
    def __init__(self,model,config,output_is_logits):
        super().__init__()
        self.model = model
        self.config = config
        self.output_is_logits = output_is_logits
        self.load_metrics(self.config)
        self.max_train_accuracy = 0
        self.max_test_accuracy = 0
        self.lr = None
        self.epoch = tf.Variable(0)
        self.batch = tf.Variable(0)

    def load_metrics(self,config):
        self.metrics_list = []
        #loss logs
        self.metrics_list.append(tf.keras.metrics.Mean(name='loss'))
        
        #accuracy logs
        if config['loss_func'] == 'categorical_crossentropy':
            self.metrics_list.append(tf.keras.metrics.CategoricalAccuracy(name='accuracy'))

    def compile(self,optimizer,loss,metrics=None):
        super().compile(optimizer=optimizer,loss=loss,metrics=metrics)
        print('Compiling Model')
        if hasattr(optimizer,'setup'):
            print('Setting up optimizer')
            self.optimizer.setup(self.model)

    def _log(items,step):
        wandb.log(items,step=self.batch)

    @tf.function
    def train_step(self, data):
        x, y = data
        if self.config['optimizer'] in ['SGD','Adam']:
            with tf.GradientTape() as tape:
                y_hat = self.model(x, training=True)  # Forward pass
                loss = self.compiled_loss(y, y_hat)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        elif self.config['optimizer'] in ['SAM_SGD','FSAM_SGD','ASAM_SGD','mSAM_SGD','lmSAM_SGD','lmSAM1_SGD','lmSAM2_SGD','NormSGD','NormSGD2','NormSGDBoxCox','NormSGD4','NormSGDFixed','CustomSGD','CustomSGDFixed']:
            loss, y_hat = self.optimizer.step(x,y,self.model,self.compiled_loss)
        elif self.config['optimizer'] in ['SAM_Metrics','FriendSAM_SGD','angleSAM_SGD']:
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
        loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE)
        with tf.GradientTape() as tape:
            #print(text_batch[i])
            y_hat = self.model(x,training=False) #get model output (softmax) [BS x num_classes]
            loss = loss_func(y,y_hat)
            selected = tf.squeeze(tf.random.categorical(tf.math.log(y_hat), 1)) #sample from the output [BS x 1]
            output = tf.gather(y_hat,selected,axis=1,batch_dims=1) #get the output for the selected class [BS x 1]
            output = tf.math.log(output) #log the output [BS x 1]
        
        j = tape.jacobian(output,self.model.trainable_variables)
        layer_sizes = [tf.reduce_sum(tf.size(v)) for v in self.model.trainable_variables] #get the size of each layer
        j = [tf.reshape(j[i],(bs,layer_sizes[i])) for i in range(len(j))] #reshape the gradient to [BS x num_layer_params x layers]
        j = tf.concat(j,axis=1) #concat the gradient over the layers [BS x num_params]
        j = tf.square(j) #square the gradient [BS x num_params]
        j = tf.reduce_sum(j,axis=1) #sum the gradient [ 1]
        return j, loss

    @tf.function
    def Get_Z_logits(self,items):
        imgs,labels = items
        bs = tf.shape(imgs)[0]
        loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)
        with tf.GradientTape() as tape:
            y_hat = self.model(imgs,training=False) #get model output  [BS x num_classes]
            loss = loss_func(labels,y_hat) #get the loss [BS x 1]
            y_hat = tf.nn.softmax(y_hat) #softmax the output [BS x num_classes]
            selected = tf.squeeze(tf.random.categorical(tf.math.log(y_hat), 1)) #sample from the output [BS x 1]
            output = tf.gather(y_hat,selected,axis=1,batch_dims=1) #get the output for the selected class [BS x 1]
            output = tf.math.log(output) #log the output [BS x 1]
        j = tape.jacobian(output,self.model.trainable_variables) #get the jacobian of the output wrt the model params [BS x num_layer_params x layers]
        layer_sizes = [tf.reduce_sum(tf.size(v)) for v in self.model.trainable_variables] #get the size of each layer
        j = [tf.reshape(j[i],(bs,layer_sizes[i])) for i in range(len(j))] #reshape the jacobian to [BS x num_layer_params x layers]
        j = tf.concat(j,axis=1) #concat the jacobian over the layers [BS x num_params]
        j = tf.square(j) #square the jacobian [BS x num_params]
        j = tf.reduce_sum(j,axis=1) #sum the jacobian [BS x 1]
        return j, loss

    @tf.function
    def Get_grads(self,items):
        x,y = items
        bs = tf.shape(x)[0]
        loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE)
        with tf.GradientTape() as tape:
            y_hat = self.model(x,training=False)
            loss = loss_func(y,y_hat)
        j = tape.jacobian(loss, self.model.trainable_variables)
        layer_sizes = [tf.reduce_sum(tf.size(v)) for v in self.model.trainable_variables] #get the size of each layer
        j = [tf.reshape(j[i],(bs,layer_sizes[i])) for i in range(len(j))] # 
        j = tf.concat(j,axis=1) #[BS x num_params] as tf
        return j, loss 

    @tf.function
    def get_params_shape(self):
        p = [tf.reshape(v,[-1]) for v in self.model.trainable_variables]
        p = tf.concat(p,axis=0)
        return p.shape[0]



def build_model(config):
    selected_model,output_is_logits = model_selector(config['model_name'],config)
    loss_func = loss_selector(config['loss_func'],config,output_is_logits)
    optimizer = optimizer_selector(config['optimizer'],config)
    metrics = metric_selector(config)

    model = Model(selected_model,config,output_is_logits)
    model.compile(optimizer=optimizer, loss=loss_func, metrics=metrics)
    callbacks = callback_selector(config,model)
    return model, callbacks

class SAM(tf.keras.optimizers.Optimizer):
    def __init__(self, base_optim, config, name="SAM", **kwargs):
        super().__init__(name, **kwargs)
        self.base_optim = base_optim
        self.rho = config['rho']  # ball size
        self.rho_decay = config['rho_decay']
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
        #tf.print(self.base_optim.lr)

    def step(self, x, y, model, loss_func):
        #compute the max step
        loss,y_hat,eps = self.max_step(model,x,y,loss_func)
        self.min_step(model,x,y,loss_func,eps)
        self.rho = self.rho * self.rho_decay
        return loss,y_hat

class mSAM(tf.keras.optimizers.Optimizer):
    #mSAM uses a small amount of data than the batch to perform the maximisation step
    def __init__(self, base_optim, config, name="mSAM", **kwargs):
        super().__init__(name, **kwargs)
        self.base_optim = base_optim
        self.rho = config['rho']  # ball size
        self.rho_decay = config['rho_decay']
        self.m_max = tf.constant(config['m'],dtype=tf.float32)
        self.m = tf.Variable(1,trainable=False,dtype=tf.int32)
        self.it = tf.Variable(0,trainable=False)
        self.bs = tf.constant(config['batch_size'])
        self.title = f"mSAM_SGD"
        self.nored_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE)
        
    def setup(self,model):
        self.group_size = tf.constant(tf.cast(self.bs/self.m,tf.int32),dtype=tf.int32)#group size
        self.count = tf.Variable(0,trainable=False)
        self.accu_grads = [tf.Variable(tf.zeros_like(v),trainable=False,dtype=tf.float32) for v in model.trainable_variables]



    @tf.function
    def get_minimisation_grad(self,model,x,y,loss_func):
        #compute grads at current point and move to the maximum in the ball
        with tf.GradientTape() as tape:
            y_hat = model(x,training=True)
            loss = self.nored_loss(y,y_hat)
            loss = tf.reduce_mean(loss)#mean the losses
        gs = tape.gradient(loss, model.trainable_variables)
        grad_norm = tf.linalg.global_norm(gs)
        eps = [(g * self.rho)/ (grad_norm + 1e-12) for g in gs]
        
        for e, var in zip(eps, model.trainable_variables):
            var.assign_add(e)

        with tf.GradientTape() as tape:
            y_hat = model(x,training=True)
            loss = loss_func(y,y_hat)
        gs = tape.gradient(loss, model.trainable_variables)

        #move back to the original point
        for e, var in zip(eps, model.trainable_variables):
            var.assign_sub(e)

        self.accu_grads = [l.assign_add(g) for l,g in zip(self.accu_grads,gs)]
        self.count.assign_add(1)
    
    def get_shape(self,x):
        return tf.shape(x)

    def step(self, x, y, model, loss_func):
        self.accu_grads = [l.assign(tf.zeros_like(l)) for l in self.accu_grads] #reset accumilated grad to 0
        self.count.assign(0)

        with tf.GradientTape() as tape:
            y_hat = model(x,training=True)
            loss = self.nored_loss(y,y_hat)
            loss = tf.reduce_mean(loss)

        if self.m > 1:
            for i in tf.range(self.m-1):
                self.get_minimisation_grad(model,x[i*self.group_size:(i+1)*self.group_size],y[i*self.group_size:(i+1)*self.group_size],loss_func)
            self.get_minimisation_grad(model,x[(i+1)*self.group_size:],y[(i+1)*self.group_size:],loss_func)
        else:
            self.get_minimisation_grad(model,x,y,loss_func)

        #average the accumulated grads
        self.accu_grads = [l.assign(tf.math.divide(l,tf.cast(self.count,dtype=tf.float32))) for l in self.accu_grads]
        #apply normal gradient step
        self.base_optim.apply_gradients(zip(self.accu_grads, model.trainable_variables))

        if tf.cast(self.it,dtype=tf.float32) > (7820.0/2.0):
            self.m.assign(tf.cast(self.m_max,dtype=tf.int32))
            self.it.assign(0)
            tf.print('m: ',self.m)
        else:
            self.it.assign_add(1)

        return loss,y_hat

class lmSAM1(tf.keras.optimizers.Optimizer):
    #mSAM uses micro batching to perform the maximisation step mico batch first then cul to highest losses in microbatch
    def __init__(self, base_optim, config, name="mSAM", **kwargs):
        super().__init__(name, **kwargs)
        self.base_optim = base_optim
        self.rho = config['rho']  # ball size
        self.rho_decay = config['rho_decay']
        self.m_max = tf.constant(config['m'],dtype=tf.float32)
        self.m = tf.Variable(1,trainable=False,dtype=tf.int32)
        self.it = tf.Variable(0,trainable=False)
        self.bs = tf.constant(config['batch_size'])
        self.cul_percent = tf.constant(0.5,dtype=tf.float32)
        self.title = f"mSAM_SGD"
        self.nored_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE)
        
        
    def setup(self,model):
        self.group_size = tf.constant(tf.cast(self.bs/self.m,tf.int32),dtype=tf.int32)#group size
        self.count = tf.Variable(0,trainable=False)
        self.accu_grads = [tf.Variable(tf.zeros_like(v),trainable=False,dtype=tf.float32) for v in model.trainable_variables]



    @tf.function
    def get_minimisation_grad(self,model,x,y,loss_func):
        #compute grads at current point and move to the maximum in the ball
        #tf.print(tf.cast(tf.cast(self.m,tf.float32)*self.cul_percent,tf.int32))
        with tf.GradientTape() as tape:
            y_hat = model(x,training=True)
            redloss = self.nored_loss(y,y_hat)
            redloss = tf.math.top_k(redloss, tf.cast(tf.cast(self.m,tf.float32)*self.cul_percent,tf.int32)).values
            loss = tf.reduce_mean(redloss)#mean the losses
        gs = tape.gradient(loss, model.trainable_variables)
        grad_norm = tf.linalg.global_norm(gs)
        eps = [(g * self.rho)/ (grad_norm + 1e-12) for g in gs]
        
        for e, var in zip(eps, model.trainable_variables):
            var.assign_add(e)

        with tf.GradientTape() as tape:
            y_hat = model(x,training=True)
            loss = loss_func(y,y_hat)
        gs = tape.gradient(loss, model.trainable_variables)

        #move back to the original point
        for e, var in zip(eps, model.trainable_variables):
            var.assign_sub(e)

        self.accu_grads = [l.assign_add(g) for l,g in zip(self.accu_grads,gs)]
        self.count.assign_add(1)
    
    def get_shape(self,x):
        return tf.shape(x)

    def step(self, x, y, model, loss_func):
        self.accu_grads = [l.assign(tf.zeros_like(l)) for l in self.accu_grads] #reset accumilated grad to 0
        self.count.assign(0)

        with tf.GradientTape() as tape:
            y_hat = model(x,training=True)
            aloss = self.nored_loss(y,y_hat)
            loss = tf.reduce_mean(aloss)
        
        #lmSAM can either be done by picking highest losses from microbatches
        # or by first picking highest losses and then microbatching

        if self.m > 1:
            for i in tf.range(self.m-1):
                self.get_minimisation_grad(model,x[i*self.group_size:(i+1)*self.group_size],y[i*self.group_size:(i+1)*self.group_size],loss_func)
            self.get_minimisation_grad(model,x[(i+1)*self.group_size:],y[(i+1)*self.group_size:],loss_func)
        else:
            self.get_minimisation_grad(model,x,y,loss_func)

        #average the accumulated grads
        #might be able to do some stats here
        self.accu_grads = [l.assign(tf.math.divide(l,tf.cast(self.count,dtype=tf.float32))) for l in self.accu_grads]
        #apply normal gradient step
        self.base_optim.apply_gradients(zip(self.accu_grads, model.trainable_variables))

        if tf.cast(self.it,dtype=tf.float32) > (7820.0/2.0):
            self.m.assign(tf.cast(self.m_max,dtype=tf.int32))
            self.it.assign(0)
            tf.print('m: ',self.m)
        else:
            self.it.assign_add(1)

        return loss,y_hat

class lmSAM2(tf.keras.optimizers.Optimizer):
    #mSAM uses micro batching to perform the maximisation step mico batch first then cul to highest losses in microbatch
    def __init__(self, base_optim, config, name="mSAM", **kwargs):
        super().__init__(name, **kwargs)
        self.base_optim = base_optim
        self.rho = config['rho']  # ball size
        self.rho_decay = config['rho_decay']
        self.m_max = tf.constant(config['m'],dtype=tf.float32)
        self.m = tf.Variable(1,trainable=False,dtype=tf.int32)
        self.it = tf.Variable(0,trainable=False)
        self.bs = tf.constant(config['batch_size'])
        self.cul_percent = tf.constant(0.5,dtype=tf.float32)
        self.title = f"mSAM_SGD"
        self.nored_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE)
        
        
    def setup(self,model):
        self.group_size = tf.constant(tf.cast(self.bs/self.m,tf.int32),dtype=tf.int32)#group size
        self.count = tf.Variable(0,trainable=False)
        self.accu_grads = [tf.Variable(tf.zeros_like(v),trainable=False,dtype=tf.float32) for v in model.trainable_variables]



    @tf.function
    def get_minimisation_grad(self,model,x,y,loss_func):
        #compute grads at current point and move to the maximum in the ball
        with tf.GradientTape() as tape:
            y_hat = model(x,training=True)
            redloss = self.nored_loss(y,y_hat)
            redloss = tf.math.top_k(loss, tf.cast(self.m*self.cul_percent,tf.int32)).values
            loss = tf.reduce_mean(redloss)#mean the losses
        gs = tape.gradient(loss, model.trainable_variables)
        grad_norm = tf.linalg.global_norm(gs)
        eps = [(g * self.rho)/ (grad_norm + 1e-12) for g in gs]
        
        for e, var in zip(eps, model.trainable_variables):
            var.assign_add(e)

        with tf.GradientTape() as tape:
            y_hat = model(x,training=True)
            loss = loss_func(y,y_hat)
        gs = tape.gradient(loss, model.trainable_variables)

        #move back to the original point
        for e, var in zip(eps, model.trainable_variables):
            var.assign_sub(e)

        self.accu_grads = [l.assign_add(g) for l,g in zip(self.accu_grads,gs)]
        self.count.assign_add(1)
    
    def get_shape(self,x):
        return tf.shape(x)

    def step(self, x, y, model, loss_func):
        self.accu_grads = [l.assign(tf.zeros_like(l)) for l in self.accu_grads] #reset accumilated grad to 0
        self.count.assign(0)

        with tf.GradientTape() as tape:
            y_hat = model(x,training=True)
            aloss = self.nored_loss(y,y_hat)
            loss = tf.reduce_mean(aloss)
        
        #lmSAM can either be done by picking highest losses from microbatches
        # or by first picking highest losses and then microbatching

        if self.m > 1:
            for i in tf.range(self.m-1):
                self.get_minimisation_grad(model,x[i*self.group_size:(i+1)*self.group_size],y[i*self.group_size:(i+1)*self.group_size],loss_func)
            self.get_minimisation_grad(model,x[(i+1)*self.group_size:],y[(i+1)*self.group_size:],loss_func)
        else:
            self.get_minimisation_grad(model,x,y,loss_func)

        #average the accumulated grads
        #might be able to do some stats here
        self.accu_grads = [l.assign(tf.math.divide(l,tf.cast(self.count,dtype=tf.float32))) for l in self.accu_grads]
        #apply normal gradient step
        self.base_optim.apply_gradients(zip(self.accu_grads, model.trainable_variables))

        if tf.cast(self.it,dtype=tf.float32) > (7820.0/2.0):
            self.m.assign(tf.cast(self.m_max,dtype=tf.int32))
            self.it.assign(0)
            tf.print('m: ',self.m)
        else:
            self.it.assign_add(1)

        return loss,y_hat

class oldlmSAM(tf.keras.optimizers.Optimizer):
    #mSAM uses a small amount of data than the batch to perform the maximisation step
    def __init__(self, base_optim, config, name="lmSAM", **kwargs):
        super().__init__(name, **kwargs)
        self.base_optim = base_optim
        self.rho = config['rho']  # ball size
        self.rho_decay = config['rho_decay']
        self.m = config['m']
        self.title = f"lmSAM_SGD"
        #currently only uses catcrossent
        self.nored_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE)
        

    @tf.function
    def max_step(self,model,x,y,loss_func):
        #compute grads at current point and move to the maximum in the ball
        with tf.GradientTape() as tape:
            y_hat = model(x,training=True)
            loss = self.nored_loss(y,y_hat)
            redloss = tf.math.top_k(loss, self.m).values #only take the m largest losses
            redloss = tf.reduce_mean(redloss)#mean the reduced losses
            gloss = tf.reduce_mean(loss)#mean of all losses
        gs = tape.gradient(redloss, model.trainable_variables)
        grad_norm = tf.linalg.global_norm(gs)
        eps = [(g * self.rho)/ (grad_norm + 1e-12) for g in gs]

        for e, var in zip(eps, model.trainable_variables):
            var.assign_add(e)
        #print('EPS: ',[e.shape for e in self.eps])
        #print('Model Trainable Variables: ',[e.shape for e in model.trainable_variables])

        #model.trainable_variables = tf.map_fn(lambda var,eps: var + eps, (model.trainable_variables, self.eps))
        return gloss,y_hat,eps
    
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
        self.rho = self.rho * self.rho_decay
        return loss,y_hat



class NormSGD_old(tf.keras.optimizers.Optimizer):
    #normalise each gradient in the batch such that higher loss does not have more power over low loss data.
    def __init__(self,base_optim,config,name="NormSGD",**kwargs):
        super().__init__(name,**kwargs)
        self.base_optim = base_optim
        self.nored_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE)
        self.scale = 0.5
        self.item_norm = tf.Variable(0.0)
        self.avg_norm = tf.Variable(0.0) # avg norm of avg batch grad

    def tf_shape(self,x):
        return tf.shape(x)

    def reduce_mult(self,x):
            y = 0
            j =0
            for xi in x:
                if j == 0:
                    y = xi
                    j+=1
                else:
                    y = y*xi
            return y

    def individual_step(self,x1,y1,model,loss_func):
        #return the normalised gradent of each item in the batch
        x1 = tf.expand_dims(x1,axis=0)
        y1 = tf.expand_dims(y1,axis=0)
        with tf.GradientTape() as tape:
            y_hat = model(x1,training=True)
            loss = loss_func(y1,y_hat)
        g = tape.gradient(loss,model.trainable_variables) #get all the gradients for the item from the batch
        g = [tf.reshape(l,[-1]) for l in g] #reshape the gradient to [num_layer_params x layers]
        g = tf.concat(g,axis=0) #concat the gradient over the layers [num_params]

        #calc the norm size of each grad and the norm direction
        g_normalise,g_norm = tf.linalg.normalize(g)
        return g_normalise,g_norm, loss,y_hat

        

    def step(self, x, y, model, loss_func):
        #get grad of each item in batch
        c = 0.0
        g,g_norm,loss,y_hat = self.individual_step(x[0],y[0],model,loss_func)
        for i in range(self.tf_shape(x)[0]):
            if i == 0:
                pass
            else:
                tg,tgn,tl,tyh = self.individual_step(x[i],y[i],model,loss_func)
                g = g+tg
                g_norm = g_norm + tgn
                loss = loss + tl
                y_hat = y_hat + tyh
            c+=1.0
        
        #avg the batch normalised grads
        g = g/c
        g_norm = g_norm/c
        loss = loss/c
        y_hat = y_hat/c
        self.item_norm.assign(tf.squeeze(g_norm))

        with tf.GradientTape() as tape:
            y_hat = model(x,training=True)
            loss = loss_func(y,y_hat)
        batch_grad = tape.gradient(loss,model.trainable_variables)
        batch_grad = [tf.reshape(l,[-1]) for l in batch_grad] #reshape the gradient to [num_layer_params x layers]
        batch_grad = tf.concat(batch_grad,axis=0)
        batch_normalise,batch_norm = tf.linalg.normalize(batch_grad)
        self.avg_norm.assign(tf.squeeze(batch_norm))

        #scale by the average norm
        g = g *g_norm


        #reshape to wights shape
        layer_shapes = [v.shape for v in model.trainable_variables]
        reshaped = [tf.zeros_like(v.shape) for v in model.trainable_variables]
        i = 0
        c = 0
        for ls in layer_shapes:
            reshaped[i] = tf.reshape(g[c:c+self.reduce_mult(ls)],ls)
            c += self.reduce_mult(ls)
            i += 1

        self.base_optim.apply_gradients(zip(reshaped,model.trainable_variables))

        return loss, y_hat

class NormSGD2_old(tf.keras.optimizers.Optimizer):
    #normalise each gradient in the batch such that higher loss does not have more power over low loss data.
    def __init__(self,base_optim,config,name="NormSGD2",**kwargs):
        super().__init__(name,**kwargs)
        self.base_optim = base_optim
        self.nored_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE)
        self.scale = 0.5
        self.item_norm = tf.Variable(0.0)
        self.avg_norm = tf.Variable(0.0) # avg norm of avg batch grad

    def tf_shape(self,x):
        return tf.shape(x)

    def reduce_mult(self,x):
        y = 0
        j =0
        for xi in x:
            if j == 0:
                y = xi
                j+=1
            else:
                y = y*xi
        return y

    def individual_step(self,x1,y1,model,loss_func):
        #return the normalised gradent of each item in the batch
        x1 = tf.expand_dims(x1,axis=0)
        y1 = tf.expand_dims(y1,axis=0)
        with tf.GradientTape() as tape:
            y_hat = model(x1,training=True)
            loss = loss_func(y1,y_hat)
        g = tape.gradient(loss,model.trainable_variables) #get all the gradients for the item from the batch
        g = [tf.reshape(l,[-1]) for l in g] #reshape the gradient to [num_layer_params x layers]
        g = tf.concat(g,axis=0) #concat the gradient over the layers [num_params]

        #calc the norm size of each grad and the norm direction
        g_normalise,g_norm = tf.linalg.normalize(g)
        return g_normalise,g_norm, loss,y_hat

        

    def step(self, x, y, model, loss_func):
        #get grad of each item in batch
        c = 0.0
        g,g_norm,loss,y_hat = self.individual_step(x[0],y[0],model,loss_func)
        for i in range(self.tf_shape(x)[0]):
            if i == 0:
                pass
            else:
                tg,tgn,tl,tyh = self.individual_step(x[i],y[i],model,loss_func)
                g = g+tg
                g_norm = g_norm + tgn
                loss = loss + tl
                y_hat = y_hat + tyh
            c+=1.0
        
        #avg the batch normalised grads
        g = g/c
        g_norm = g_norm/c
        loss = loss/c
        y_hat = y_hat/c
        self.item_norm.assign(tf.squeeze(g_norm))

        #scale by the average norm of the average batch not component average
        with tf.GradientTape() as tape:
            y_hat = model(x,training=True)
            loss = loss_func(y,y_hat)
        batch_grad = tape.gradient(loss,model.trainable_variables)
        batch_grad = [tf.reshape(l,[-1]) for l in batch_grad] #reshape the gradient to [num_layer_params x layers]
        batch_grad = tf.concat(batch_grad,axis=0)
        batch_normalise,batch_norm = tf.linalg.normalize(batch_grad)
        g = g *batch_norm
        self.avg_norm.assign(tf.squeeze(batch_norm))


        #reshape to wights shape
        layer_shapes = [v.shape for v in model.trainable_variables]
        reshaped = [tf.zeros_like(v.shape) for v in model.trainable_variables]
        i = 0
        c = 0
        for ls in layer_shapes:
            reshaped[i] = tf.reshape(g[c:c+self.reduce_mult(ls)],ls)
            c += self.reduce_mult(ls)
            i += 1

        self.base_optim.apply_gradients(zip(reshaped,model.trainable_variables))

        return loss, y_hat

#NormSGD can use dir and scale of both sgd and normalised grads also does fixed scaling
class NormSGD(tf.keras.optimizers.Optimizer):
    #dir(SGD) scale(norm)
    def __init__(self,base_optim,config,name="NormSGD",**kwargs):
        super().__init__(name,**kwargs)
        self.base_optim = base_optim
        self.scale = config['norm_scale']
        self.direction = config['direction']
        self.batch_norm = tf.Variable(0.0) 


    def tf_shape(self,x):
        return tf.shape(x)

    def reduce_mult(self,x):
        y = 0
        j =0
        for xi in x:
            if j == 0:
                y = xi
                j+=1
            else:
                y = y*xi
        return y

    def individual_step(self,x1,y1,model,loss_func):
        #return the normalised gradent of each item in the batch
        x1 = tf.expand_dims(x1,axis=0)
        y1 = tf.expand_dims(y1,axis=0)
        with tf.GradientTape() as tape:
            y_hat = model(x1,training=True)
            loss = loss_func(y1,y_hat)
        g = tape.gradient(loss,model.trainable_variables) #get all the gradients for the item from the batch
        g = [tf.reshape(l,[-1]) for l in g] #reshape the gradient to [num_layer_params x layers]
        g = tf.concat(g,axis=0) #concat the gradient over the layers [num_params]
        return g, loss,y_hat

    def step(self, x, y, model, loss_func):
        #get grad of each item in batch
        c = 0.0
        y_hats = tf.TensorArray(tf.float32, size=self.tf_shape(x)[0])
        g_sgd,loss,y_hat = self.individual_step(x[0],y[0],model,loss_func)
        g_norm = tf.linalg.normalize(g_sgd)[0]
        y_hats = y_hats.write(0,tf.squeeze(y_hat))
        for i in range(self.tf_shape(x)[0]):
            if i == 0:
                pass
            else:
                tg,tl,tyh = self.individual_step(x[i],y[i],model,loss_func)
                g_sgd = g_sgd+tg #sum the sgd grads
                g_norm = g_norm + tf.linalg.normalize(tg)[0]
                loss = loss + tl
                y_hats = y_hats.write(i,tf.squeeze(tyh))
            c+=1.0
        
        #get final norms and direction
        g_sgd = g_sgd/c #average the sgd grads
        g_norm = g_norm/c #average the normalised grads
        scale_sgd = tf.norm(g_sgd) #get the scale of the sgd grads
        scale_norm = tf.norm(g_norm) #get the scale of the normalised grads

        #combine dir and scale
        if isinstance(self.scale, str):
            if self.scale == 'sgd':
                scale = scale_sgd
            elif self.scale == 'norm':
                scale = scale_norm
            else:
                scale = 100.0 #indicates a probelm
        else:
            scale = self.scale #fixed scaling

        self.batch_norm.assign(scale)


        if self.direction == 'sgd':
            g = tf.linalg.normalize(g_sgd)[0] * scale
        elif self.direction == 'norm':
            g = tf.linalg.normalize(g_norm)[0] * scale
        else:
            g = tf.linalg.normalize(g_norm)[0] * scale #shouldnt end up here

        loss = loss/c
        y_hats = y_hats.stack()

        #reshape to wights shape
        layer_shapes = [v.shape for v in model.trainable_variables]
        reshaped = [tf.zeros_like(v.shape) for v in model.trainable_variables]
        i = 0
        c = 0
        for ls in layer_shapes:
            reshaped[i] = tf.reshape(g[c:c+self.reduce_mult(ls)],ls)
            c += self.reduce_mult(ls)
            i += 1

        self.base_optim.apply_gradients(zip(reshaped,model.trainable_variables))

        return loss, y_hats

class CustomSGD(tf.keras.optimizers.Optimizer):
    def __init__(self,base_optim,config,name="CustomSGD",**kwargs):
        super().__init__(name,**kwargs)
        self.batch_norm = tf.Variable(0.0)
        self.base_optim = base_optim
    
    def step(self,x,y,model,loss_func):
        with tf.GradientTape() as tape:
            y_hat = model(x,training=True)
            loss = loss_func(y,y_hat)
        g = tape.gradient(loss,model.trainable_variables)

        flat_g = [tf.reshape(v,[-1]) for v in g]
        flat_g = tf.concat(flat_g,axis=0)
        
        self.batch_norm.assign(tf.norm(flat_g))


        self.base_optim.apply_gradients(zip(g,model.trainable_variables))
        return loss, y_hat

class NormSGDBoxCox(tf.keras.optimizers.Optimizer):
    #fix the skew in the norm distribution of the gradients
    # - This is the dir(boxcox nomalised grads) scale()
    def __init__(self,base_optim,config,name="NormSGDboxcox",**kwargs):
        super().__init__(name,**kwargs)
        self.base_optim = base_optim
        self.nored_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE)
        self.scale = config['norm_scale']
        self.direction = config['direction']
        self.batch_norm = tf.Variable(0.0) #norm of gradient use in update

        #these will be updated at the end of the epoch
        self.lam = tf.Variable(0.0)
        
        self.recorded_points = 10000
        self.norm_list = tf.Variable(tf.zeros([self.recorded_points]),trainable=False)
        self.norm_index = tf.Variable(0) #set to zero at end of epoch


    def tf_shape(self,x):
        return tf.shape(x)

    def reduce_mult(self,x):
        y = 0
        j =0
        for xi in x:
            if j == 0:
                y = xi
                j+=1
            else:
                y = y*xi
        return y

    def individual_step(self,x1,y1,model,loss_func):
        #return the normalised gradent of each item in the batch
        x1 = tf.expand_dims(x1,axis=0)
        y1 = tf.expand_dims(y1,axis=0)
        with tf.GradientTape() as tape:
            y_hat = model(x1,training=True)
            loss = loss_func(y1,y_hat)
        g = tape.gradient(loss,model.trainable_variables) #get all the gradients for the item from the batch
        g = [tf.reshape(l,[-1]) for l in g] #reshape the gradient to [num_layer_params x layers]
        g = tf.concat(g,axis=0) #concat the gradient over the layers [num_params]
        return g, loss,y_hat

    def boxcox(self,x):
        if self.lam == 0:
            #this is traditionally the log function but our lam is 0 on first iteration
            return x
        else:
            return (x**self.lam - 1)/self.lam

    def set_lambda(self,lam):
        #called at eoe
        self.lam.assign(lam)
        self.norm_index.assign(0)


    def update_norm_list(self,norm,index):
        if index < self.recorded_points:
            self.norm_list[index].assign(norm)
            self.norm_index.assign_add(1)


    def step(self, x, y, model, loss_func):
        #get grad of each item in batch
        g_sgd,loss,y_hat = self.individual_step(x[0],y[0],model,loss_func)
        g_normalised, g_norm = tf.linalg.normalize(g_sgd)
        self.update_norm_list(g_norm,self.norm_index) #add the first item to the norm list
        g_norm = self.boxcox(g_norm) #transform norm with boxcox
        g_boxcox = g_normalised * g_norm # dir * updated norm

        y_hats = tf.TensorArray(tf.float32, size=self.tf_shape(x)[0])
        y_hats = y_hats.write(0,tf.squeeze(y_hat))

        c = 0.0
        for i in range(self.tf_shape(x)[0]):
            if i == 0:
                pass
            else:
                tg_sgd,tl,tyh = self.individual_step(x[i],y[i],model,loss_func)
                tgnormalised,tgnorm = tf.linalg.normalize(tg_sgd)
                self.update_norm_list(tgnorm,self.norm_index)
                tgnorm = self.boxcox(tgnorm)
                tg = tgnormalised * tgnorm # += dir * updated norm
                g_sgd = g_sgd+tg_sgd #sum the sgd grads
                g_boxcox = g_boxcox + tg #sum the boxcox grads
                loss = loss + tl
                y_hats = y_hats.write(i,tf.squeeze(tyh))
            c+=1.0

        g_sgd = g_sgd/c #average the sgd grads
        g_boxcox = g_boxcox/c #average the boxcox grads
        loss = loss/c
        y_hats = y_hats.stack()

        if isinstance(self.scale, str):
            if self.scale == 'sgd':
                scale = tf.norm(g_sgd)
            elif self.scale == 'boxcox':
                scale = tf.norm(g_boxcox)
            else:
                scale = 100.0
        else:
            scale = self.scale

        self.batch_norm.assign(scale)

        if self.direction == 'sgd':
            g = tf.linalg.normalize(g_sgd)[0] * scale
        elif self.direction == 'boxcox':
            g = tf.linalg.normalize(g_boxcox)[0] * scale
        else:
            g = tf.linalg.normalize(g_boxcox)[0] * scale

        #reshape to wights shape
        layer_shapes = [v.shape for v in model.trainable_variables]
        reshaped = [tf.zeros_like(v.shape) for v in model.trainable_variables]
        i = 0
        c = 0
        for ls in layer_shapes:
            reshaped[i] = tf.reshape(g[c:c+self.reduce_mult(ls)],ls)
            c += self.reduce_mult(ls)
            i += 1

        self.base_optim.apply_gradients(zip(reshaped,model.trainable_variables))

        return loss, y_hats

class NormSGDFixed(tf.keras.optimizers.Optimizer):
    #normalise each gradient in the batch such that higher loss does not have more power over low loss data.
    def __init__(self,base_optim,config,name="NormSGDFixed",**kwargs):
        super().__init__(name,**kwargs)
        self.base_optim = base_optim
        self.scale = config['norm_scale']
        self.batch_norm = tf.Variable(0.0) # avg of norms of items

    def tf_shape(self,x):
        return tf.shape(x)

    def reduce_mult(self,x):
        y = 0
        j =0
        for xi in x:
            if j == 0:
                y = xi
                j+=1
            else:
                y = y*xi
        return y

    def individual_step(self,x1,y1,model,loss_func):
        #return the normalised gradent of each item in the batch
        x1 = tf.expand_dims(x1,axis=0)
        y1 = tf.expand_dims(y1,axis=0)
        with tf.GradientTape() as tape:
            y_hat = model(x1,training=True)
            loss = loss_func(y1,y_hat)
        g = tape.gradient(loss,model.trainable_variables) #get all the gradients for the item from the batch
        g = [tf.reshape(l,[-1]) for l in g] #reshape the gradient to [num_layer_params x layers]
        g = tf.concat(g,axis=0) #concat the gradient over the layers [num_params]

        #calc the norm size of each grad and the norm direction
        g_normalise,g_norm = tf.linalg.normalize(g)
        return g_normalise,g_norm, loss,y_hat

    def step(self, x, y, model, loss_func):
        #get grad of each item in batch
        c = 0.0
        y_hats = tf.TensorArray(tf.float32, size=self.tf_shape(x)[0])
        g,g_norm,loss,y_hat = self.individual_step(x[0],y[0],model,loss_func)
        y_hats = y_hats.write(0,tf.squeeze(y_hat))
        for i in range(self.tf_shape(x)[0]):
            if i == 0:
                pass
            else:
                tg,tgn,tl,tyh = self.individual_step(x[i],y[i],model,loss_func)
                g = g+tg 
                loss = loss + tl
                #concat the predictions
                y_hats = y_hats.write(i,tf.squeeze(tyh))
            c+=1.0
            
        y_hats = y_hats.stack()

        g,_ = tf.linalg.normalize(g) #direction with length 1
        loss = loss/c
        self.batch_norm.assign(self.scale)

        g = g *self.scale #dir(norm) scale(fixed)

        #reshape to wights shape
        layer_shapes = [v.shape for v in model.trainable_variables]
        reshaped = [tf.zeros_like(v.shape) for v in model.trainable_variables]
        i = 0
        c = 0
        for ls in layer_shapes:
            reshaped[i] = tf.reshape(g[c:c+self.reduce_mult(ls)],ls)
            c += self.reduce_mult(ls)
            i += 1

        self.base_optim.apply_gradients(zip(reshaped,model.trainable_variables))

        return loss, y_hats

class CustomSGDFixed(tf.keras.optimizers.Optimizer):
    def __init__(self,base_optim,config,name="CustomSGDFixed",**kwargs):
        super().__init__(name,**kwargs)
        self.batch_norm = tf.Variable(0.0)
        self.base_optim = base_optim
        self.scale = config['norm_scale']
    
    def reduce_mult(self,x):
        y = 0
        j =0
        for xi in x:
            if j == 0:
                y = xi
                j+=1
            else:
                y = y*xi
        return y
    
    def step(self,x,y,model,loss_func):
        with tf.GradientTape() as tape:
            y_hat = model(x,training=True)
            loss = loss_func(y,y_hat)
        g = tape.gradient(loss,model.trainable_variables)

        flat_g = [tf.reshape(v,[-1]) for v in g]
        flat_g = tf.concat(flat_g,axis=0)

        flat_g = tf.linalg.normalize(flat_g)[0] #scale =1 just direction
        flat_g = flat_g * self.scale #scale by fixed amount
        self.batch_norm.assign(self.scale)

        #reshape to wights shape
        layer_shapes = [v.shape for v in model.trainable_variables]
        reshaped = [tf.zeros_like(v.shape) for v in model.trainable_variables]
        i = 0
        c = 0
        for ls in layer_shapes:
            reshaped[i] = tf.reshape(flat_g[c:c+self.reduce_mult(ls)],ls)
            c += self.reduce_mult(ls)
            i += 1

        self.base_optim.apply_gradients(zip(reshaped,model.trainable_variables))
        return loss, y_hat



class FriendSAM(tf.keras.optimizers.Optimizer):
    #mSAM uses a small amount of data than the batch to perform the maximisation step
    def __init__(self, base_optim, config, name="lmSAM", **kwargs):
        super().__init__(name, **kwargs)
        self.base_optim = base_optim
        self.rho = config['rho']  # ball size
        self.rho_decay = config['rho_decay']
        self.m = config['m']
        self.title = f"lmSAM_SGD"
        #currently only uses catcrossent
        self.nored_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE)
        self.lam = 0.99
        tf.print('in FriendSAM init')
        
    def setup(self,model):
        self.m = [tf.Variable(tf.zeros_like(v),trainable=False) for v in model.trainable_variables]
        tf.print('in FriendSAM setup')

    @tf.function
    def max_step(self,model,x,y,loss_func):
        #compute grads at current point and move to the maximum in the ball
        with tf.GradientTape() as tape:
            y_hat = model(x,training=True)
            loss = self.nored_loss(y,y_hat)
            gloss = tf.reduce_mean(loss)#mean of all losses
        gs = tape.gradient(gloss, model.trainable_variables)
        grad_norm = tf.linalg.global_norm(gs)
        eps = [(g * self.rho)/ (grad_norm + 1e-12) for g in gs]

        for m,g in zip(self.m,gs):
            m.assign(self.lam * m + (1-self.lam) * g)
        #self.m = [self.lam * m + (1-self.lam) * g for m,g in zip(self.m,gs)]

        for e, var in zip(eps, model.trainable_variables):
            var.assign_add(e)
        #print('EPS: ',[e.shape for e in self.eps])
        #print('Model Trainable Variables: ',[e.shape for e in model.trainable_variables])

        #model.trainable_variables = tf.map_fn(lambda var,eps: var + eps, (model.trainable_variables, self.eps))
        return gloss,y_hat,eps
    
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
        #calc the batch noise component of the gradient
        gs_flat = tf.concat([tf.reshape(v,[-1]) for v in gs],axis=0)
        m_flat = tf.concat([tf.reshape(v,[-1]) for v in self.m],axis=0)
        angle = tf.tensordot(gs_flat,m_flat,axes=1)/(tf.norm(gs_flat)*tf.norm(m_flat))
        noise = [gb - angle * gf for gb,gf in zip(gs,self.m)]

        self.base_optim.apply_gradients(zip(noise, model.trainable_variables))

    def step(self, x, y, model, loss_func):
        #compute the max step
        loss,y_hat,eps = self.max_step(model,x,y,loss_func)
        self.min_step(model,x,y,loss_func,eps)
        self.rho = self.rho * self.rho_decay
        return loss,y_hat

class ASAM(tf.keras.optimizers.Optimizer):
    #Adaptive SAM
    def __init__(self, base_optim, config, name="ASAM", **kwargs):
        super().__init__(name, **kwargs)
        self.base_optim = base_optim
        self.rho = config['rho']  # ball size
        self.rho_decay = config['rho_decay']
        self.title = f"ASAM_SGD"

    @tf.function
    def max_step(self,model,x,y,loss_func):
        #compute grads at current point and move to the maximum in the ball
        with tf.GradientTape() as tape:
            y_hat = model(x,training=True)
            loss = loss_func(y,y_hat)
        gs = tape.gradient(loss, model.trainable_variables)
        #tf.print('Gradients: ',[e for e in gs])
        #flatten model weights
        #flat_weights = tf.concat([tf.reshape(v,[-1]) for v in model.trainable_variables],axis=0)
        t_w = [tf.math.abs(l) for l in model.trainable_variables]
        t_w_l = [tf.math.multiply(tl,gl) for tl,gl in zip(t_w,gs)]
        t_norm = tf.norm(tf.concat([tf.reshape(t,[-1]) for t in t_w_l],axis=0))

        eps = [self.rho * tf.math.multiply(t,t_l) for t,t_l in zip(t_w,t_w_l)]
        eps = [tf.math.divide(e,t_norm) for e in eps]
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
        self.rho = self.rho * self.rho_decay
        return loss,y_hat

class FSAM(tf.keras.optimizers.Optimizer):
    #Fisher SAM
    def __init__(self, base_optim, config, name="FSAM", **kwargs):
        super().__init__(name, **kwargs)
        self.base_optim = base_optim
        self.rho = config['rho']  # ball size
        self.rho_decay = config['rho_decay']
        self.title = f"FSAM_SGD"
        self.print_flag = True
        self.r = 1 #reg term

    #@tf.function
    def max_step(self,model,x,y,loss_func):
        #compute grads at current point and move to the maximum in the ball
        #compute fisher
        # with tf.GradientTape() as tape:
        #     y_hat = model(x,training=True)
        #     loss = loss_func(y,y_hat)
        # gs = tape.gradient(loss, model.trainable_variables)
        # i_fisher = [tf.square(i) for i in gs]
        # i_fisher = [tf.math.reciprocal(tf.add(i*self.r,1)) for i in i_fisher]
        with tf.GradientTape() as tape:
            y_hat = model(x,training=True)
            selected = tf.squeeze(tf.random.categorical(tf.math.log(y_hat), 1)) #sample from the output [BS x 1]
            output = tf.gather(y_hat,selected,axis=1,batch_dims=1) #get the output for the selected class [BS x 1]
            output = tf.math.log(output) #log the output [BS x 1]
        j = tape.jacobian(output,model.trainable_variables) #get the jacobian of the output wrt the model params [BS x num_layer_params x layers]
        j = [tf.reduce_mean(i,axis=0) for i in j]#average over the batch
        j = [tf.square(i) for i in j]
        i_fisher = [tf.math.reciprocal(tf.add(i*self.r,1)) for i in j]
        
        with tf.GradientTape() as tape:
            y_hat = model(x,training=True)
            loss = loss_func(y,y_hat)
        gs = tape.gradient(loss, model.trainable_variables)


        eps_u = [self.rho * tf.math.multiply(tl,gl) for tl,gl in zip(i_fisher,gs)]
        eps_l = [tf.math.sqrt(tf.math.multiply(tf.math.multiply(tl,gl),gl)) for tl,gl in zip(i_fisher,gs)]
        eps = [tf.math.divide(u,l+1.e-12) for u,l in zip(eps_u,eps_l)]

        # if self.print_flag:
        #     self.print_flag = False
        #     tf.print('y_hat: ',y_hat)
        #     tf.print('Loss: ',loss)
        #     tf.print('Params: ',[e.shape for e in model.trainable_variables])
        #     tf.print('Gradients: ',[e for e in gs])
        #     tf.print('Fisher Information: ',[e for e in i_fisher])
        #     tf.print('EPS_u: ',[e for e in eps_u])
        #     tf.print('EPS_l: ',[e for e in eps_l])
        #     tf.print('EPS: ',[e for e in eps])

        
        
        for e, var in zip(eps, model.trainable_variables):
            var.assign_add(e)

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
        self.rho = self.rho * self.rho_decay

        return loss,y_hat

class LookSAM(tf.keras.optimizers.Optimizer):
    def __init__(self, base_optim, config, k=5, alpha=0.5, name="LookSAM", **kwargs):
        super().__init__(name, **kwargs)
        self.base_optim = base_optim
        self.rho = config['rho']
        self.k = k #update frequence of sharpness grad
        self.k_current = 0
        self.title = f"LookSAM"

    def calc_g(self,x,y,model,loss_func):
        with tf.GradientTape() as tape:
            y_hat = model(x,training=True)
            loss = loss_func(y,y_hat)
        return tape.gradient(loss,model.trainable_variables)
    
    def calc_gs(self,x,y,g,model,loss_func):
        grad_norm = tf.linalg.global_norm(g)
        eps = [(g * self.rho)/ (grad_norm + 1e-12) for g in gs]
        for e, var in zip(eps, model.trainable_variables):
            var.assign_add(e)
        with tf.GradientTape() as tape:
            y_hat = model(x,training=True)
            loss = loss_func(y,y_hat)
        gs = tape.gradient(loss, model.trainable_variables)
        angle = tf.tensordot(g,gs,axes=1)/(tf.norm(g)*tf.norm(gs))
        gv = gs - tf.norm(gs)*tf.tensordot(angle, g/tf.norm(g),axes=1)

        for e, var in zip(eps, model.trainable_variables):
            var.assign_sub(e)
        return gs,gv

    def step(self,x,y,model,loss_func):
        g = calc_g(x,y,model,loss_func) #grad from current point
        if self.k_current % self.k == 0:
            g_s,self.g_v = calc_gs(x,y,g,model,loss_func) #calc the grad from advanced point
        else:
            g_s = g + alpha*(tf.norm(g)/tf.norm(self.g_v))*self.g_v
        self.k_current += 1
        self.base_optim.apply_gradients(zip(g_s,model.trainable_variables))

class angleSAM(tf.keras.optimizers.Optimizer):
    def __init__(self, base_optim, config, name="angleSAM", **kwargs):
        super().__init__(name, **kwargs)
        self.base_optim = base_optim
        self.rho = config['rho']
        self.title = f"angleSAM"
        self.bs = tf.constant(config['batch_size'])
        self.angles = tf.Variable(tf.zeros((self.bs,)),trainable=False)
        self.gradcount = tf.Variable(0,trainable=False, dtype=tf.float32)

    def setup(self,model):
        self.accgrad = [tf.Variable(tf.zeros_like(v),trainable=False) for v in model.trainable_variables]
        

    def tf_shape(self,x):
        return tf.shape(x)
    
    @tf.function
    def cos_sim(self,x,y,g,loss_func,model):
        #x is single data point
        #g is the mean gradient
        with tf.GradientTape() as tape:
            y_hat = model(x,training=True)
            loss = loss_func(y,y_hat)
        item_g = tape.gradient(loss, model.trainable_variables)
        fitem_g = tf.concat([tf.reshape(v,[-1]) for v in item_g],axis=0)
        angle = tf.tensordot(g,fitem_g,axes=1)/(tf.norm(g)*tf.norm(fitem_g))

        if angle > 0:
            i = 0
            for l in self.accgrad:
                l.assign_add(item_g[i])
                i += 1
            self.gradcount.assign_add(1)

    @tf.function
    def max_step(self,model,x,y,loss_func):
        #compute grads at current point and move to the maximum in the ball
        with tf.GradientTape() as tape:
            y_hat = model(x,training=True)
            loss = loss_func(y,y_hat)
        gs = tape.gradient(loss, model.trainable_variables)
        g_flat = tf.concat([tf.reshape(v,[-1]) for v in gs],axis=0) #flat avg grad

        self.gradcount.assign(0)
        self.accgrad = [l.assign(tf.zeros_like(l)) for l in self.accgrad] #reset accumilated grad to 0
        for i in tf.range(self.tf_shape(x)[0]):
            self.cos_sim(tf.expand_dims(x[i],axis=0),tf.expand_dims(y[i],axis=0),g_flat,loss_func,model)
        
        #average the accumulated grads
        for l in self.accgrad:
            l.assign(tf.math.divide(l,self.gradcount))
    

        grad_norm = tf.linalg.global_norm(self.accgrad)
        eps = [(g * self.rho)/ (grad_norm + 1e-12) for g in self.accgrad]

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
        #tf.print(self.base_optim.lr)

    def step(self, x, y, model, loss_func):
        #compute the max step
        loss,y_hat,eps = self.max_step(model,x,y,loss_func)
        self.min_step(model,x,y,loss_func,eps)
        return loss,y_hat

    

class SAM_Metrics(tf.keras.optimizers.Optimizer):
    def __init__(self, record_its,base_optim, config, name="SAM_Metrics", **kwargs):
        super().__init__(name, **kwargs)
        self.record_its = record_its
        self.its = tf.Variable(0,trainable=False)
        self.base_optim = base_optim
        self.rho = config['rho']
        self.title = f"SAM_Metrics"
        self.bs = tf.constant(config['batch_size'])
        self.g_diff = tf.Variable(0.0,trainable=False)
        self.g_s_diff = tf.Variable(0.0,trainable=False)
        self.g_v_diff = tf.Variable(0.0,trainable=False)
        self.angles = tf.Variable(tf.zeros((config['batch_size'],)),trainable=False)
        self.nored_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE)
        #self.old_g_flat = tf.Variable(tf.zeros((315722,)),trainable=False) #need a way to get model params
        #self.old_g_s_flat = tf.Variable(tf.zeros((315722,)),trainable=False)
        #self.old_g_v_flat = tf.Variable(tf.zeros((315722,)),trainable=False)
    
    def setup(self,model):
        p = [tf.reshape(v,[-1]) for v in model.trainable_variables]
        p = tf.concat(p,axis=0)
        self.old_g_flat = tf.Variable(tf.zeros((p.shape[0],)),trainable=False)
        self.old_g_s_flat = tf.Variable(tf.zeros((p.shape[0],)),trainable=False)
        self.old_g_v_flat = tf.Variable(tf.zeros((p.shape[0],)),trainable=False)

    def tf_shape(self,x):
        return tf.shape(x)

    @tf.function
    def cos_sim(self,x,y,g,loss_func,model):
        #x is single data point
        #g is the mean gradient
        with tf.GradientTape() as tape:
            y_hat = model(x,training=True)
            loss = loss_func(y,y_hat)
        item_g = tape.gradient(loss, model.trainable_variables)
        item_g = tf.concat([tf.reshape(v,[-1]) for v in item_g],axis=0)
        return tf.tensordot(g,item_g,axes=1)/(tf.norm(g)*tf.norm(item_g))

    @tf.function
    def og_grad(self,model,x,y,loss_func):
        with tf.GradientTape() as tape:
            y_hat = model(x,training=True)
            loss = loss_func(y,y_hat)
        g = tape.gradient(loss, model.trainable_variables)

        #print('g: ',[e.shape for e in g])
        g_flat = tf.concat([tf.reshape(v,[-1]) for v in g],axis=0) #flat avg grad

        s = self.tf_shape(x)[0]
        if s == self.bs:
            for i in tf.range(self.bs):
                angle = self.cos_sim(tf.expand_dims(x[i],axis=0),tf.expand_dims(y[i],axis=0),g_flat,loss_func,model)
                self.angles[i].assign(angle)
        
    
        #return the angle between the mean gradient and the batch gradient
        return g



    @tf.function
    def sam_grad(self,model,x,y,loss_func,g):
        grad_norm = tf.linalg.global_norm(g)
        eps = [(g_i * self.rho)/ (grad_norm + 1e-12) for g_i in g]
        for e, var in zip(eps, model.trainable_variables):
            var.assign_add(e)
        with tf.GradientTape() as tape:
            y_hat = model(x,training=True)
            loss = loss_func(y,y_hat)
        g_s = tape.gradient(loss, model.trainable_variables)

        for e, var in zip(eps, model.trainable_variables):
            var.assign_sub(e)
        return g_s,loss,y_hat

    @tf.function
    def v_grad(self,grad,g_sgrad):
        g_flat = tf.concat([tf.reshape(v,[-1]) for v in grad],axis=0)
        g_s_flat = tf.concat([tf.reshape(v,[-1]) for v in g_sgrad],axis=0)
        angle = tf.tensordot(g_flat,g_s_flat,axes=1)/tf.norm(g_flat)*tf.norm(g_s_flat)
        g_v_flat = g_s_flat - tf.norm(g_s_flat)*tf.tensordot(angle, g_flat/tf.norm(g_flat),axes=0)
        return g_flat,g_s_flat,g_v_flat


    @tf.function
    def update_its(self):
        self.its.assign_add(1)
    
    @tf.function
    def new_vs_old_grads(self,g_flat,g_s_flat,g_v_flat):
        self.g_diff.assign(tf.norm(self.old_g_flat - g_flat))
        self.g_s_diff.assign(tf.norm(self.old_g_s_flat - g_s_flat))
        self.g_v_diff.assign(tf.norm(self.old_g_v_flat - g_v_flat))


    @tf.function
    def buildoldgrads(self,g_flat,g_s_flat,g_v_flat):
        self.old_g_flat.assign(g_flat)
        self.old_g_s_flat.assign(g_s_flat)
        self.old_g_v_flat.assign(g_v_flat)

    
    @tf.function
    def step0(self,x,y,model,loss_func):
        g = self.og_grad(model,x,y,loss_func)
        g_s,loss,y_hat = self.sam_grad(model,x,y,loss_func,g)

        g_flat,g_s_flat,g_v_flat = self.v_grad(g,g_s)

        self.buildoldgrads(g_flat,g_s_flat,g_v_flat)
        self.base_optim.apply_gradients(zip(g_s, model.trainable_variables))
        self.update_its()
        return loss,y_hat
    @tf.function
    def recordstep(self,x,y,model,loss_func):
        g = self.og_grad(model,x,y,loss_func)
        g_s,loss,y_hat = self.sam_grad(model,x,y,loss_func,g)

        g_flat,g_s_flat,g_v_flat = self.v_grad(g,g_s)

        self.new_vs_old_grads(g_flat,g_s_flat,g_v_flat)
        self.buildoldgrads(g_flat,g_s_flat,g_v_flat)
        self.base_optim.apply_gradients(zip(g_s, model.trainable_variables))
        self.update_its()
        return loss,y_hat

    @tf.function
    def choosestep(self,x,y,model,loss_func):
        loss, y_hat = tf.cond(tf.math.floormod(self.its,self.record_its) == 0, lambda: self.recordstep(x,y,model,loss_func),lambda: self.norecordstep(x,y,model,loss_func))
        return loss,y_hat

    @tf.function
    def norecordstep(self,x,y,model,loss_func):
        g = self.og_grad(model,x,y,loss_func)
        g_s,loss,y_hat = self.sam_grad(model,x,y,loss_func,g)
        self.base_optim.apply_gradients(zip(g_s, model.trainable_variables))
        self.update_its()
        return loss,y_hat

    def step(self,x,y,model,loss_func):
        loss,y_hat = tf.cond(self.its == 0,lambda: self.step0(x,y,model,loss_func),lambda: self.choosestep(x,y,model,loss_func))
        return loss,y_hat




class LRMetric(tf.keras.metrics.Metric):
    def __init__(self, optimizer, name='LR', **kwargs):
        super().__init__(name=name, **kwargs)
        self.lr = self.add_variable(
            shape=(),
            initializer='zeros',
            name='lr'
        )
        if hasattr(optimizer, 'base_optim'):
            self.optimizer = optimizer.base_optim
        else:
            self.optimizer = optimizer

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.lr.assign(self.optimizer.lr)

    def result(self):
        return self.lr

class SAMMetric(tf.keras.metrics.Metric):
    def __init__(self, optimizer, name='SAMMetric', **kwargs):
        super().__init__(name=name, **kwargs)
        self.gdiff = self.add_variable(
            shape=(),
            initializer='zeros',
            name='gdiff'
        )
        self.gsdiff = self.add_variable(
            shape=(),
            initializer='zeros',
            name='gsdiff'
        )
        self.gvdiff = self.add_variable(
            shape=(),
            initializer='zeros',
            name='gvdiff'
        )
        self.optimizer = optimizer
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.gdiff.assign(self.optimizer.g_diff)
        self.gsdiff.assign(self.optimizer.g_s_diff)
        self.gvdiff.assign(self.optimizer.g_v_diff)
    def result(self):
        return self.gdiff,self.gsdiff,self.gvdiff



def fixed_lrschedule(epoch,lr,config):
    return lr

def percentage_step_decay_lrschedule(epoch,lr,config):
    change_epochs = [int(config['epochs']*i) for i in config['lr_decay_params']['lr_decay_epochs_percent']]
    tf.print('Change Epochs: ',change_epochs)
    tf.print('Epoch: ',epoch)
    if int(epoch) in change_epochs:
        lr = lr * config['lr_decay_params']['lr_decay_rate']
    return lr

def cosine_decay_lrschedule(step,lr,config):
    step = min(step,config['lr_decay_params']['decay_steps'])
    cosine_decay = 0.5 * (1 +tf.math.cos(np.pi * step / config['lr_decay_params']['decay_steps']))
    decayed = (1 - config['lr_decay_params']['alpha']) * cosine_decay + config['lr_decay_params']['alpha']
    return config['lr'] * decayed



def callback_selector(config,model):
    class CustomLRScheduler(tf.keras.callbacks.Callback):
        #sets the LR for the model
        def __init__(self,lr_schedule,config):
            super().__init__()
            self.lr_schedule = lr_schedule
            self.config = config
            self.local_lr = 0
            self.epochs = 0
            self.batches = 0

        def on_batch_begin(self, batch, logs=None):
            if self.config['lr_decay_type'] in ['cosine_decay']:
                #check if the optimzer has a base_optim attribute
                if hasattr(self.model.optimizer, 'base_optim'):
                    lr = float(tf.keras.backend.get_value(self.model.optimizer.base_optim.lr))
                    new_lr = self.lr_schedule((self.epochs * self.batches)+batch,lr,self.config)
                    self.local_lr = new_lr
                    tf.keras.backend.set_value(self.model.optimizer.base_optim.lr, new_lr)

                else:
                    lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
                    new_lr = self.lr_schedule((self.epochs * self.batches)+batch,lr,self.config)
                    self.local_lr = new_lr
                    tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
            #tf.print('Learning Rate: ',self.local_lr)
            if self.epochs == 0:
                self.batches += 1
                
        
        def on_epoch_begin(self, epoch, logs=None):
            if self.config['lr_decay_type'] in ['fixed','percentage_step_decay']:
                #check if the optimzer has a base_optim attribute
                if hasattr(self.model.optimizer, 'base_optim'):
                    lr = float(tf.keras.backend.get_value(self.model.optimizer.base_optim.lr))
                    new_lr = self.lr_schedule(epoch,lr,self.config)
                    self.local_lr = new_lr
                    tf.keras.backend.set_value(self.model.optimizer.base_optim.lr, new_lr)

                else:
                    lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
                    new_lr = self.lr_schedule(epoch,lr,self.config)
                    self.local_lr = new_lr
                    tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
        
        def on_epoch_end(self, epoch, logs=None):
            #wandb.log({'lr':self.local_lr},commit=True)
            #tf.print('Learning Rate: ',self.local_lr)
            self.epochs += 1
            print('Epoch: ',epoch)
            print('Batches: ',self.batches)
            wandb.log({'lr':self.local_lr},step=tf.keras.backend.get_value(self.model.batch))
            tf.print('Learning Rate: ',self.local_lr)

    class CustomSAMCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()

        def on_batch_end(self, batch, logs=None):
            its = int(tf.keras.backend.get_value(self.model.optimizer.its))
            record_its = int(self.model.optimizer.record_its)
            if its % record_its == 0 and its != 0:
                gdiff = float(tf.keras.backend.get_value(self.model.optimizer.g_diff))
                gsdiff = float(tf.keras.backend.get_value(self.model.optimizer.g_s_diff))
                gvdiff = float(tf.keras.backend.get_value(self.model.optimizer.g_v_diff))
                wandb.log({'gdiff':gdiff,'gsdiff':gsdiff,'gvdiff':gvdiff},step=tf.keras.backend.get_value(self.model.batch))
                angles = tf.keras.backend.get_value(self.model.optimizer.angles)
                wandb.log({'angles':angles},step=tf.keras.backend.get_value(self.model.batch))
    
    class BestAccuracy(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best_acc = 0
            self.best_val_acc = 0
        
        def on_test_end(self,logs=None):
            #record the best test acc
            print(tf.keras.backend.get_value(self.model.metrics[1].result()))
            curr_acc = float(tf.keras.backend.get_value(self.model.metrics[1].result()))
            #curr_val_acc = float(tf.keras.backend.get_value(self.model.metrics_list.val_accuracy))

            if curr_acc > self.best_acc:
                self.best_acc = curr_acc
            #if curr_val_acc > self.best_val_acc:
                #self.best_val_acc = curr_val_acc

            wandb.log({'best_acc':self.best_acc},step=tf.keras.backend.get_value(self.model.batch))
        
    class NormCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
        def on_batch_end(self,batch,logs=None):
            bn = tf.keras.backend.get_value(self.model.batch)
            if hasattr(self.model.optimizer, 'item_norm'):
                wandb.log({'item_norm':float(tf.keras.backend.get_value(self.model.optimizer.item_norm))},step=bn)
            if hasattr(self.model.optimizer, 'avg_norm'):
                wandb.log({'avg_norm':float(tf.keras.backend.get_value(self.model.optimizer.avg_norm))},step=bn)
            if hasattr(self.model.optimizer, 'batch_norm'):
                wandb.log({'batch_norm':float(tf.keras.backend.get_value(self.model.optimizer.batch_norm))},step=bn)

    class EpochUpdate(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.epoch = tf.Variable(0)
            self.batch = tf.Variable(0)
        
        def on_epoch_end(self,epoch,logs=None):
            if hasattr(self.model.optimizer, 'epoch'):
                tf.keras.backend.set_value(self.model.optimizer.epoch,epoch)
            
            if hasattr(self.model, 'epoch'):
                tf.keras.backend.set_value(self.model.epoch,self.epoch)
        
        def on_batch_end(self,batch,logs=None):
            if hasattr(self.model.optimizer, 'batch'):
                tf.keras.backend.set_value(self.model.optimizer.batch,batch)
            self.batch.assign_add(1)

            if hasattr(self.model, 'batch'):
                tf.keras.backend.set_value(self.model.batch,self.batch)

            
            
    #this defiens any callback for the fit function
    callbacks = []
    #learning rate changes
    if config['lr_decay_type'] == 'fixed':
        lr_callback = CustomLRScheduler(fixed_lrschedule,config)
        callbacks.append(lr_callback)
    elif config['lr_decay_type'] == 'percentage_step_decay':
        lr_callback = CustomLRScheduler(percentage_step_decay_lrschedule,config)
        callbacks.append(lr_callback)
    elif config['lr_decay_type'] == 'cosine_decay':
        print('Cosine Decay Setup')
        lr_callback = CustomLRScheduler(cosine_decay_lrschedule,config)
        callbacks.append(lr_callback)

    if config['optimizer'] == 'SAM_Metrics':
        sam_callback = CustomSAMCallback()
        callbacks.append(sam_callback)

    best_accuracy = BestAccuracy()
    callbacks.append(best_accuracy)

    norms = NormCallback()
    callbacks.append(norms)

    epochupdate = EpochUpdate()
    callbacks.append(epochupdate)
    print('Callbacks: ',callbacks)
    
    return callbacks
    
    
        

def metric_selector(config):
    metrics = []
    if config['loss_func'] == 'categorical_crossentropy':
        metrics.append(tf.keras.metrics.CategoricalAccuracy())

    return metrics


def loss_selector(loss_name, config, output_is_logits=False):
    if loss_name == 'categorical_crossentropy':
        return tf.keras.losses.CategoricalCrossentropy(from_logits=output_is_logits)




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

    def build_preact_resnet(x,block_type,blocks_per_layer,n_cls,model_width=64,REG=0):
        #modified to tf from https://github.com/tml-epfl/understanding-sam/blob/main/deep_nets/models.py#L231
        kaiming_normal = keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal')

        def preact_block(x,planes,stride,downsample=None,name=None):
            #maybe need to do some more work on the "act_function" call with softplus?
            out = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn1')(x)
            out = tf.keras.layers.ReLU(name=f'{name}.relu1')(out)
            if stride != 1 or inplanes != planes:
                shortcut = tf.keras.layers.Conv2D(filters=planes, kernel_size=1, strides=stride, use_bias=False, kernel_initializer=kaiming_normal,kernel_regularizer=keras.regularizers.l2(REG), name=f'{name}.shortcut.0')(out)
            else:
                shortcut = x
            out = tf.keras.layers.Conv2D(filters=planes, kernel_size=3, strides=stride, use_bias=False, kernel_initializer=kaiming_normal,kernel_regularizer=keras.regularizers.l2(REG), name=f'{name}.conv1')(out)
            out = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn2')(out)
            out = tf.keras.layers.ReLU(name=f'{name}.relu2')(out)
            out = tf.keras.layers.Conv2D(filters=planes, kernel_size=3, strides=1, use_bias=False, kernel_initializer=kaiming_normal,kernel_regularizer=keras.regularizers.l2(REG), name=f'{name}.conv2')(out)
            out = tf.keras.layers.Add(name=f'{name}.add')([out, shortcut])
            return out

        def make_layer(x, block_type,planes,num_blocks,stride,name=None):
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                if block_type == 'basic_block':
                    x = basic_block(x,planes,stride,downsample=None,name=name)
                    inplanes = planes * basic_block.expansion
                elif block_type == 'preact_block':
                    x = preact_block(x,planes,stride,downsample=None,name=name)
                    inplanes = planes * 1
            return x

        def resnet(x, block_type, blocks_per_layer, num_classes, model_width):
            #maybe a norm layer here --->
            x = tf.keras.layers.Conv2D(filters=model_width, kernel_size=3, strides=1, use_bias=False, kernel_initializer=kaiming_normal,kernel_regularizer=keras.regularizers.l2(REG), name='conv1')(x)

            x = make_layer(x, block_type, model_width, blocks_per_layer[0], 1, name='layer1')
            x = make_layer(x, block_type, 2*model_width, blocks_per_layer[1], 2, name='layer2')
            x = make_layer(x, block_type, 4*model_width, blocks_per_layer[2], 2, name='layer3')
            x = make_layer(x, block_type, 8*model_width, blocks_per_layer[3], 2, name='layer4')

            x = tf.keras.layers.BatchNormalization(name='bn1')(x)
            x = tf.keras.layers.ReLU(name='relu1')(x)
            x = tf.keras.layers.AveragePooling2D(4,name='avgpool')(x)
            x = tf.keras.layers.Flatten(name='flatten')(x)
            return x
        inplanes = model_width
        return resnet(x, block_type, blocks_per_layer, n_cls, model_width)

    def resnetV2(x,stackwise_features,stackwise_blocks,stackwise_strides,block_type,num_classes,REG=0):
        #https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-build-a-resnet-from-scratch-with-tensorflow-2-and-keras.md
        #x = input tensor
        #stackwise_features =[64,128,256,512]
        #stackwise_blocks =[2,2,2,2]
        #stackwise_strides=[1,2,2,2]
        #so: {[3x3,64],[3x3,64]}+{[3x3,128],[3x3,128]}+{[3x3,256],[3x3,256]}+{[3x3,512],[3x3,512]}
        #block type = string of either 'basic_block' or 'block'
        #shortcut type = string of either 'identity' or 'projection'
        initializer = tf.keras.initializers.HeNormal()
        # if block_type == 'basic_block':
        #     block_fn = BasicBlock
        # elif block_type == 'block':
        #     block_fn = PreActBlock
        # else:
        #     raise ValueError('Block type not recognised')
        def BasicBlock(x,filters,kernel_size=3,stride=1,conv_shortcut=False):
            preact = tf.keras.layers.BatchNormalization(axis=3)(x)
            preact = tf.keras.layers.ReLU()(preact)

            if conv_shortcut:
                shortcut = tf.keras.layers.Conv2D(filters,kernel_size=1,strides=stride,kernel_initializer=initializer)(preact)
            else:
                shortcut = tf.keras.layers.MaxPooling2D(1,strides=stride)(x) if stride > 1 else x

            x = tf.keras.layers.Conv2D(filters,kernel_size,strides=1,padding='same',kernel_initializer=initializer,use_bias=False)(preact)
            x = tf.keras.layers.BatchNormalization(axis=3)(x)
            x = tf.keras.layers.ReLU()(x)

            x = tf.keras.layers.Conv2D(filters,kernel_size,strides=stride,padding='same',kernel_initializer=initializer,use_bias=False)(x)
            x = tf.keras.layers.Add()([shortcut,x])
            return x

        def Stack(x,filters,blocks,stride,dilations,block_type,first_shortcut):
            #this is collection of blocks
            x = block_fn(x,filters,match_filter_size=first_shortcut)
            for block in range(2,blocks):
                x = block_fn(x,filters)
            x = block_fn(x,filters,stride=stride)
            return x

        def ResBlocks(x):
            #stackwise_dialations = [1]*len(stackwise_features)
            #filter_size = init_feature_maps
            for layer_group in range(len(stackwise_features)):
                for block in range(n):
                    if layer_group > 0 and block == 0:
                        #filter_size = filter_size * 2
                        #x = BasicBlock(x,filter_size,match_filter_size=True)
                        x = BasicBlock(x,stackwise_features[layer_group],match_filter_size=True)
                    else:
                        x = BasicBlock(x,filter_size)
            return x


        def resnet(x):
            x = tf.keras.layers.Conv2D(64,kernel_size=7,strides=2,padding='same',kernel_initializer=initializer)(x)
            x = tf.keras.layers.MaxPooling2D(pool_size=3,strides=2,padding='same')(x)
            x = ResBlocks(x)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Flatten()(x)
            output = tf.keras.layers.Dense(num_classes,activation='softmax',kernel_initializer=initializer)(x)
            return output
        return resnet(x)

    #https://github.com/meng1994412/ResNet_from_scratch
    def mengResNet56(x,num_classes,REG=0):
        def residual_module(x,features,stride,chanDim,reduce=False,REG=0,bnEps=2e-5,bnMom=0.9):
            shortcut = x
            bn1 = tf.keras.layers.BatchNormalization(axis=chanDim,epsilon=bnEps,momentum=bnMom)(x)
            act1 = tf.keras.layers.Activation('relu')(bn1)
            conv1 = tf.keras.layers.Conv2D(features,(3,3),strides=stride,padding='same',kernel_regularizer=keras.regularizers.l2(REG))(act1)
            bn2 = tf.keras.layers.BatchNormalization(axis=chanDim,epsilon=bnEps,momentum=bnMom)(conv1)
            act2 = tf.keras.layers.Activation('relu')(bn2)
            conv2 = tf.keras.layers.Conv2D(features,(3,3),strides=(1,1),padding='same',kernel_regularizer=keras.regularizers.l2(REG))(act2)
            if reduce:
                shortcut = tf.keras.layers.Conv2D(features,(1,1),strides=stride,padding='same',kernel_regularizer=keras.regularizers.l2(REG))(act1)
            x = tf.keras.layers.add([conv2,shortcut])
            return x

    class ResBlock(tf.keras.layers.Layer):
        def __init__(self,filters,downsample=False,strides=(2,2),reg=0,bnEps=2e-5,bnMom=0.9,bottleneck=True):
            super(ResBlock,self).__init__()
            self.bn1 = tf.keras.layers.BatchNormalization(epsilon=bnEps,momentum=bnMom)
            self.act1 = tf.keras.layers.Activation('relu')
            if bottleneck:
                self.conv1 = tf.keras.layers.Conv2D(int(filters*0.25),(1,1),strides=(1,1),padding='same',use_bias=False,kernel_initializer='he_normal',kernel_regularizer=keras.regularizers.l2(reg))
            else:
                self.conv1 = tf.keras.layers.Conv2D(filters,(1,1),strides=(1,1),padding='same',use_bias=False,kernel_initializer='he_normal',kernel_regularizer=keras.regularizers.l2(reg))
            self.bn2 = tf.keras.layers.BatchNormalization(epsilon=bnEps,momentum=bnMom)
            self.act2 = tf.keras.layers.Activation('relu')
            if bottleneck:
                self.conv2 = tf.keras.layers.Conv2D(int(filters*0.25),(3,3),strides=strides,padding='same',use_bias=False,kernel_initializer='he_normal')
            else:
                self.conv2 = tf.keras.layers.Conv2D(filters,(3,3),strides=strides,padding='same',use_bias=False,kernel_initializer='he_normal')
            self.bn3 = tf.keras.layers.BatchNormalization(epsilon=bnEps,momentum=bnMom)
            self.act3 = tf.keras.layers.Activation('relu')
            self.conv3 = tf.keras.layers.Conv2D(filters,(1,1),strides=(1,1),padding='same',use_bias=False,kernel_initializer='he_normal')

            self.downsample = downsample
            self.shortcut = tf.keras.layers.Conv2D(filters,(1,1),strides=strides,padding='same',use_bias=False,kernel_initializer='he_normal')
            self.strides = strides
            self.filters = filters

            
        def __call__(self,x,training=False):
            shortcut = x
            x = self.bn1(x)
            x = self.act1(x)
            out = self.conv1(x)
            
            out = self.bn2(out)
            out = self.act2(out)
            out = self.conv2(out)

            out = self.bn3(out)
            out = self.act3(out)
            out = self.conv3(out)
            if self.downsample:
                shortcut = self.shortcut(x)
            out += shortcut
            return out

    class CifarResNet(tf.keras.Model):
        #ResNet with bottelnecking and pre-activation
        #https://stackoverflow.com/questions/71057776/why-does-my-resnet56-implementation-have-less-accuracy-than-in-the-original-pape
        #https://arxiv.org/pdf/1512.03385.pdf
        #https://github.com/meng1994412/ResNet_from_scratch/blob/master/pipeline/nn/conv/resnet.py
        
        def __init__(self,num_classes,filters=[64,128,256],n=9,REG=0,bnEps=2e-5,bnMom=0.9,bottleneck=True):
            super(CifarResNet,self).__init__()
            self.filters = filters
            self.n = n
            self.REG = REG
            self.bottleneck = bottleneck

            self.bn1 = tf.keras.layers.BatchNormalization(epsilon=bnEps,momentum=bnMom)
            self.conv1 = tf.keras.layers.Conv2D(filters[0],(3,3),strides=(1,1),padding='same',use_bias=False,kernel_initializer='he_normal',kernel_regularizer=keras.regularizers.l2(REG))

            self.block1 = self._make_layer(0)
            self.block2 = self._make_layer(1)
            self.block3 = self._make_layer(2)

            self.bn2 = tf.keras.layers.BatchNormalization(epsilon=bnEps,momentum=bnMom)
            self.act1 = tf.keras.layers.Activation('relu')
            self.ap = tf.keras.layers.AveragePooling2D((8,8))
            self.flatten = tf.keras.layers.Flatten()
            self.dense = tf.keras.layers.Dense(num_classes,activation='softmax',kernel_regularizer=keras.regularizers.l2(REG))

        def _make_layer(self,layer_id):
            seq_model = tf.keras.Sequential()
            if layer_id == 0:
                seq_model.add(ResBlock(self.filters[0],strides=(1,1),downsample=True,reg=self.REG,bottleneck=self.bottleneck))
                for i in range(self.n-1):
                    seq_model.add(ResBlock(self.filters[0],strides=(1,1),downsample=False,reg=self.REG,bottleneck=self.bottleneck))
            elif layer_id == 1:
                seq_model.add(ResBlock(self.filters[1],strides=(2,2),downsample=True,reg=self.REG,bottleneck=self.bottleneck))
                for i in range(self.n-1):
                    seq_model.add(ResBlock(self.filters[1],strides=(1,1),downsample=False,reg=self.REG,bottleneck=self.bottleneck))
            elif layer_id == 2:
                seq_model.add(ResBlock(self.filters[2],strides=(2,2),downsample=True,reg=self.REG,bottleneck=self.bottleneck))
                for i in range(self.n-1):
                    seq_model.add(ResBlock(self.filters[2],strides=(1,1),downsample=False,reg=self.REG,bottleneck=self.bottleneck))
            return seq_model

        def call(self,inputs,training=False):
            out = self.bn1(inputs)
            out = self.conv1(out)

            out = self.block1(out)
            out = self.block2(out)
            out = self.block3(out)
            
            out = self.bn2(out)
            out = self.act1(out)
            out = self.ap(out)
            out = self.flatten(out)
            out = self.dense(out)
            return out
            
    class WideBasic(tf.keras.layers.Layer):
        def __init__(self,in_planes,planes,stride,droprate,REG=0.0):
            super(WideBasic,self).__init__()
            self.bn1 = tf.keras.layers.BatchNormalization(scale=True,center=True)
            self.act1 = tf.keras.layers.Activation('relu')
            self.conv1 = tf.keras.layers.Conv2D(planes, kernel_size=3, strides=1, padding='same', use_bias=True, kernel_regularizer=keras.regularizers.l2(REG))
            self.do1 = tf.keras.layers.Dropout(droprate)
            self.bn2 = tf.keras.layers.BatchNormalization(scale=True,center=True)
            self.act2 = tf.keras.layers.Activation('relu')
            self.conv2 = tf.keras.layers.Conv2D(planes, kernel_size=3, strides=stride, padding='same', use_bias=True, kernel_regularizer=keras.regularizers.l2(REG))

            if stride != 1 or in_planes != planes:
                self.shortcut = tf.keras.layers.Conv2D(planes, kernel_size=1, strides=stride, use_bias=True, kernel_regularizer=keras.regularizers.l2(REG))
            else:
                self.shortcut = tf.keras.layers.Lambda(lambda x: x)

        def call(self,x):
            out = self.bn1(x)
            skip = self.shortcut(x)

            out = self.act1(out)
            out = self.conv1(out)
            out = self.do1(out)
            out = self.bn2(out)
            out = self.act2(out)
            out = self.conv2(out)
            out += skip
            return out

    class WideResNet(tf.keras.layers.Layer):
        def __init__(self,block,num_classes,depth=28,widen_factor=10,activation='relu',droprate=0.0,REG=0.0):
            super(WideResNet,self).__init__()
            self.n = (depth-4)//6
            self.k = widen_factor
            self.n_stages = [16, 16*self.k, 32*self.k, 64*self.k]

            #init layers
            self.conv1 = tf.keras.layers.Conv2D(self.n_stages[0], kernel_size=3, strides=1, padding='same', use_bias=True, kernel_regularizer=keras.regularizers.l2(REG))
            self.layer1 = self._make_layer(block, self.n_stages[0], self.n_stages[1], self.n, 1, droprate,REG)
            self.layer2 = self._make_layer(block, self.n_stages[1], self.n_stages[2], self.n, 2, droprate,REG)
            self.layer3 = self._make_layer(block, self.n_stages[2], self.n_stages[3], self.n, 2, droprate,REG)
            self.bn = tf.keras.layers.BatchNormalization(scale=True,center=True)
            self.act = tf.keras.layers.Activation('relu')
            self.avgpool = tf.keras.layers.AveragePooling2D(8)
            self.reshape = tf.keras.layers.Flatten()
            self.dense = tf.keras.layers.Dense(num_classes,activation='softmax')

        def _make_layer(self,block, in_planes, out_planes, num_blocks, stride, droprate,REG=0.0):
            strides = [stride] + [1]*(num_blocks-1)
            seq_model = tf.keras.Sequential()
            for stride in strides:
                seq_model.add(block(in_planes,out_planes,stride,droprate,REG))
                in_planes = out_planes
            return seq_model
        
        def call(self,x):
            out = self.conv1(x)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.bn(out)
            out = self.act(out)
            out = self.avgpool(out)
            out = self.reshape(out)
            out = self.dense(out)
            return out

    class PreActBlock(tf.keras.layers.Layer):
        expansion = 1
        def __init__(self,in_planes,planes, stride=1,droprate=0.0):
            super(PreActBlock,self).__init__()

            self.bn1 = tf.keras.layers.BatchNormalization()
            self.conv1 = tf.keras.layers.Conv2D(planes, kernel_size=3, strides=stride, padding='same', use_bias=False,kernel_initializer='he_normal')
            self.bn2 = tf.keras.layers.BatchNormalization()
            #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=not learnable_bn)
            self.conv2 = tf.keras.layers.Conv2D(planes, kernel_size=3, strides=1, padding='same', use_bias=False,kernel_initializer='he_normal')        
            self.dropout = tf.keras.layers.Dropout(droprate)
            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = tf.keras.Sequential(tf.keras.layers.Conv2D(self.expansion*planes, kernel_size=1, strides=stride, use_bias=False,kernel_initializer='he_normal'))        
            print('blockinit')

        def call(self,x,training=False):
            out = tf.nn.relu(self.bn1(x))
            out = self.dropout(out,training=training)
            shortcut = self.shortcut(out) if hasattr(self,'shortcut') else x
            out = self.conv1(out)
            out = tf.nn.relu(self.bn2(out))
            out = self.dropout(out,training=training)
            out = self.conv2(out)
            out += shortcut
            return out

    class PreActResNet(tf.keras.Model):
        def __init__(self,block,blocks_per_layer,num_classes,droprate=0.0):
            super(PreActResNet,self).__init__()
            self.in_planes = 64
            self.droprate = droprate

            self.conv1 = tf.keras.layers.Conv2D(64,kernel_size=3, strides=1, padding='same', use_bias=False,kernel_initializer='he_normal')        
            self.layer1 = self._make_layer(block, 64, blocks_per_layer[0], stride=1)
            self.layer2 = self._make_layer(block, 128, blocks_per_layer[1], stride=2)
            self.layer3 = self._make_layer(block, 256, blocks_per_layer[2], stride=2)                                           
            self.layer4 = self._make_layer(block, 512, blocks_per_layer[3], stride=2)
            self.linear = tf.keras.layers.Dense(num_classes,activation='softmax',kernel_initializer='he_normal')
            self.AP = tf.keras.layers.AveragePooling2D(4)                                                               
            self.flat = tf.keras.layers.Flatten()
            print('Main init sucesess')
        def _make_layer(self, block, planes, num_blocks, stride):                                                               
            strides = [stride] + [1]*(num_blocks-1)
            layer = tf.keras.Sequential()
            for stride in strides:
                layer.add(block(self.in_planes,planes, stride=stride,droprate=self.droprate))
                self.in_planes = planes * block.expansion
            return layer

        def call(self,x):
            out = self.conv1(x)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.AP(out)
            out = self.flat(out)
            out = self.linear(out)
            return out

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
    elif config['model_name'] == "CNN5":
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=config['img_size'], kernel_initializer=initialiser),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
            tf.keras.layers.Conv2D(64,3,activation='relu', kernel_initializer=initialiser),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(128,3,activation='relu', kernel_initializer=initialiser),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128,activation='relu', kernel_initializer=initialiser),
            tf.keras.layers.Dense(config['num_classes'], kernel_initializer=initialiser),
            tf.keras.layers.Softmax()
        ])
        output_is_logits = False

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
        
    elif config['model_name'] == "ResNet18":
        #build resnet18 model
        inputs = keras.Input(shape=config['img_size'])
        outputs = build_resnet(inputs,[2,2,2,2],config['num_classes'],REG=0)
        model = keras.Model(inputs, outputs)
        output_is_logits = False

    elif config['model_name'] == "ResNet18V2":
        inputs = keras.Input(shape=config['img_size'])
        outputs = resnetV2(inputs,[64,128,256,512],[2,2,2,2],[1,2,2,2],'basic_block',config['num_classes'],REG=0)
        model = keras.Model(inputs, outputs)
        output_is_logits = False

    elif config['model_name'] == 'CifarResNet56':
        model = CifarResNet(10,n=9,REG=config['weight_reg'],filters=[64,128,256],bottleneck=True)
        model.build(input_shape=(None,32,32,3))
        output_is_logits = False

    elif config['model_name'] == "PA_ResNet18":
        #this is a [64,128,256,512] resnet18 with preactivations
        inputs = keras.Input(shape=config['img_size'])
        model = PreActResNet(PreActBlock,[2,2,2,2],config['num_classes'],droprate=config['dropout'])(inputs)
        model.build(input_shape=None + config['img_size'])
        output_is_logits = False

    elif config['model_name'] == "WRN28-10":
        inputs = keras.Input(shape=config['img_size'])
        #block,num_classes,depth=28,widen_factor=10,activation='relu',droprate=0.0,REG=0.0
        outputs = WideResNet(WideBasic,config['num_classes'],depth=28,widen_factor=10,activation='relu',droprate=0.0,REG=config['weight_reg'])(inputs)
        model = keras.Model(inputs, outputs)
        output_is_logits = False


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

def optimizer_selector(optimizer_name,config):
    if optimizer_name == 'SGD':
        optim= tf.keras.optimizers.SGD(learning_rate=config['lr'],momentum=config['momentum'])
    elif optimizer_name == 'NormSGD':
        optim= NormSGD(tf.keras.optimizers.SGD(learning_rate=config['lr'],momentum=config['momentum']),config)
    elif optimizer_name == 'NormSGDBoxCox':
        optim = NormSGDBoxCox(tf.keras.optimizers.SGD(learning_rate=config['lr'],momentum=config['momentum']),config)
    #elif optimizer_name == 'NormSGD3':
    #    optim = NormSGD3(tf.keras.optimizers.SGD(learning_rate=config['lr'],momentum=config['momentum']),config)
    elif optimizer_name == 'CustomSGD':
        optim= CustomSGD(tf.keras.optimizers.SGD(learning_rate=config['lr'],momentum=config['momentum']),config)
    elif optimizer_name == 'CustomSGDFixed':
        optim= CustomSGDFixed(tf.keras.optimizers.SGD(learning_rate=config['lr'],momentum=config['momentum']),config)
    #elif optimizer_name == 'NormSGD4':
    #    optim= NormSGD4(tf.keras.optimizers.SGD(learning_rate=config['lr'],momentum=config['momentum']),config)
    #elif optimizer_name == 'NormSGDFixed':
    #    optim= NormSGDFixed(tf.keras.optimizers.SGD(learning_rate=config['lr'],momentum=config['momentum']),config)
    elif optimizer_name == 'Adam':
        optim= tf.keras.optimizers.Adam(learning_rate=config['lr'])
    elif optimizer_name == 'SAM_SGD':
        optim= SAM(tf.keras.optimizers.SGD(learning_rate=config['lr'],momentum=config['momentum']),config)
    elif optimizer_name == 'FSAM_SGD':
        optim= FSAM(tf.keras.optimizers.SGD(learning_rate=config['lr'],momentum=config['momentum']),config)
    elif optimizer_name == 'ASAM_SGD':
        optim= ASAM(tf.keras.optimizers.SGD(learning_rate=config['lr'],momentum=config['momentum']),config)
    elif optimizer_name == 'mSAM_SGD':
        optim= mSAM(tf.keras.optimizers.SGD(learning_rate=config['lr'],momentum=config['momentum']),config)
    elif optimizer_name == 'lmSAM_SGD':
        optim= lmSAM(tf.keras.optimizers.SGD(learning_rate=config['lr'],momentum=config['momentum']),config)
    elif optimizer_name == 'lmSAM1_SGD':
        optim= lmSAM1(tf.keras.optimizers.SGD(learning_rate=config['lr'],momentum=config['momentum']),config)
    elif optimizer_name == 'lmSAM2_SGD':
        optim= lmSAM2(tf.keras.optimizers.SGD(learning_rate=config['lr'],momentum=config['momentum']),config)
    elif optimizer_name == 'SAM_Metrics':
        optim= SAM_Metrics(5,tf.keras.optimizers.SGD(learning_rate=config['lr'],momentum=config['momentum']),config)
    elif optimizer_name == 'FriendSAM_SGD':
        optim= FriendSAM(tf.keras.optimizers.SGD(learning_rate=config['lr'],momentum=config['momentum']),config)
    elif optimizer_name == 'angleSAM_SGD':
        optim= angleSAM(tf.keras.optimizers.SGD(learning_rate=config['lr'],momentum=config['momentum']),config)
    else:
        print('Optimizer not recognised')
        return None
    
    return optim
    
    