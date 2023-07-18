#This is a repository for the models used in the project 
#it should also contain the hyperparameters for the runs

import tensorflow as tf
import wandb   
from tensorflow import keras
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
        if self.config.loss_func == 'categorical_crossentropy':
            self.loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=self.output_is_logits,label_smoothing=self.config.label_smoothing)
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
        print('INIT: Model: ',self.config.model_name)
        #define the model
        #TODO add more models
        #TODO Add model init seeds
        if self.config.model_name == "CNN":
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=self.dataset_info.features['image'].shape),
                tf.keras.layers.MaxPool2D(),
                tf.keras.layers.Conv2D(64,3,activation='relu'),
                tf.keras.layers.MaxPool2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128,activation='relu'),
                tf.keras.layers.Dense(self.dataset_info.features['label'].num_classes,activation='softmax')
            ])
            self.output_is_logits = False
        
        else:
            print('Model not recognised')

        self.model.build(input_shape=self.dataset_info.features['image'].shape + (1,))

    def model_compile(self):
        self.model.summary()
        self.model.compile(optimizer=self.optimizer,loss=self.loss_func,metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])

    def lr_schedule(self,epoch,init=False):
        #this needs to define the learning rate schedule
        #THIS NEEDS LOOKING AT
        if self.config.lr_decay_type == 'step':
            self.lr = self.config.lr * self.config.lr_decay**(epoch//self.config.lr_decay_end)
        if self.config.lr_decay_type == 'fixed':
            self.lr = self.config.lr
        elif self.config.lr_decay_type == 'cosine':
            self.lr = self.config.lr * 0.5 * (1 + tf.math.cos((epoch/self.config.epochs)*3.141592653589793))
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
        if self.test_results[0] > self.max_acc:
            self.max_acc = self.test_results[0]
            self.early_stop_count = 0
        else:
            self.early_stop_count += 1

        if self.early_stop_count >= self.config.early_stop:
            return True
        else:
            return False

    def log_metrics(self):
        wandb.log({'train_loss':self.train_loss_metric.result(),'train_acc':self.train_acc_metric.result(),'train_prec':self.train_prec_metric.result(),'train_rec':self.train_rec_metric.result(),'test_loss':self.test_results[0],'test_acc':self.test_results[1],'test_prec':self.test_results[2],'test_rec':self.test_results[3],'lr':self.lr},step=self.epoch_num)

    def calc_FIM(self,dataset):
        #this needs to define the FIM
        #calc fim diag
        print('FIM: Calculating FIM')
        t = time.time()
        data_count = 0
        msq = 0
        lower_lim = np.min([self.config.record_FIM_n_data_points,dataset.total_train_data_points])
        for i in range(lower_lim):
            img,_ = dataset.__getitem__(i,training=False,return_loss=False)
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
        img = tf.expand_dims(img,0)
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
    def get_batch_loss(self,imgs,labels,training=False):
        preds = self.model(imgs,training=training)
        loss = self.loss_func(labels,preds)
        return loss

