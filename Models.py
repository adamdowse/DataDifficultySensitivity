#This is a repository for the models used in the project 
#it should also contain the hyperparameters for the runs

import tensorflow as tf
import wandb   



class Models():
    def __init__(self,config,dataset_info):
        #this needs to define hyperparams as well as the model
        self.epoch_num = 0
        self.batch_num = 0
        self.config = config
        optimizer_init()
        loss_func_init()
        metrics_init()
        model_init()
        self.lr_schedule(0,True)
        self.max_acc = 0
        
        self.dataset_info = dataset_info


        def optimizer_init(self):
            #this needs to define the optimizer
            if config['optimizer'] == 'Adam':
                self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
            elif config['optimizer'] == 'SGD':
                self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.config.lr)
            elif config['optimizer'] == 'Momentum':
                self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.config.lr,momentum=self.config.momentum)
            else:
                print('Optimizer not recognised')    

        def loss_func_init(self):
            #this needs to define the loss function
            if config.loss == 'CategoricalCrossentropy':
                self.loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=self.output_is_logits,label_smoothing=config.label_smoothing)
            else:
                print('Loss not recognised')

        def metrics_init(self):
            self.train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
            self.train_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
            self.train_prec_metric = tf.keras.metrics.Precision(name='train_precision')
            self.train_rec_metric = tf.keras.metrics.Recall(name='train_recall')

            self.test_loss_metric = tf.keras.metrics.Mean(name='test_loss')
            self.test_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
            self.test_prec_metric = tf.keras.metrics.Precision(name='test_precision')
            self.test_rec_metric = tf.keras.metrics.Recall(name='test_recall')

        def model_init(self):
            #this needs to define the model
            if config.model_name == "CNN":
                self.model = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=dataset_info.input_shape),
                    tf.keras.layers.MaxPool2D(),
                    tf.keras.layers.Conv2D(64,3,activation='relu'),
                    tf.keras.layers.MaxPool2D(),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(128,activation='relu'),
                    tf.keras.layers.Dense(dataset_info.num_classes,activation='softmax')
                ])
                self.output_is_logits = False

            self.model.build(input_shape=dataset_info.input_shape+(1,))
            self.model.summary()
            self.model.compile(optimizer=self.optimizer,loss=self.loss_func,metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])

        
    def lr_schedule(self,epoch,init=False):
        #this needs to define the learning rate schedule
        #THIS NEEDS LOOKING AT
        if self.config.lr_decay_type == 'step':
            self.lr = self.config.lr * self.config.lr_decay**(epoch//self.config.lr_decay_end)
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

        self.test_loss_metric.reset_states()
        self.test_acc_metric.reset_states()
        self.test_prec_metric.reset_states()
        self.test_rec_metric.reset_states()

        return
    
    def early_stop(self):
        #this needs to define the early stop
        #returns true if early stop is triggered
        #check test accuracy
        if self.test_acc_metric.result() > self.max_acc:
            self.max_acc = self.test_acc_metric.result()
            self.early_stop_count = 0
        else:
            self.early_stop_count += 1

        if self.early_stop_count >= self.config.early_stop:
            return True
        else:
            return False

    def log_metrics(self):
        wandb.log({'train_loss':self.train_loss_metric.result(),'train_acc':self.train_acc_metric.result(),'train_prec':self.train_prec_metric.result(),'train_rec':self.train_rec_metric.result(),'test_loss':self.test_loss_metric.result(),'test_acc':self.test_acc_metric.result(),'test_prec':self.test_prec_metric.result(),'test_rec':self.test_rec_metric.result(),'lr':self.lr},step=self.epoch_num)

    def calc_FIM(self,dataset):
        #this needs to define the FIM
        #calc fim diag
        data_count = 0
        msq = 0
        while data_count < self.config.record_FIM_n_data_points:
            img,label = dataset.get_single_train_data()
            data_count += 1
            #calc sum of squared grads for a data point and class square rooted
            z = self.Get_Z(self.model,img)
            if data_count == 1:
                mean = z
            delta = z - mean
            mean += delta / (data_count+1) #Welford_cpp from web
            msq += delta * (z - mean)

        self.train_FIM = mean
        self.train_FIM_var = msq/(data_count-1)

    def calc_complex_FIM(self,dataset):
        #calc fim diag
        data_count = 0
        msq = 0
        while data_count < self.config.record_FIM_n_data_points:
            img,label = dataset.get_low_loss_single_train_data()
            data_count += 1
            #calc sum of squared grads for a data point and class square rooted
            z = self.Get_Z(self.model,img)
            if data_count == 1:
                mean = z
            delta = z - mean
            mean += delta / (data_count+1) #Welford_cpp from web
            msq += delta * (z - mean)

        self.low_loss_train_FIM = mean
        self.low_loss_train_FIM_var = msq/(data_count-1)

        data_count = 0
        msq = 0
        while data_count < self.config.record_FIM_n_data_points:
            img,label = dataset.get_high_loss_single_train_data()
            data_count += 1
            #calc sum of squared grads for a data point and class square rooted
            z = self.Get_Z(self.model,img)
            if data_count == 1:
                mean = z
            delta = z - mean
            mean += delta / (data_count+1) #Welford_cpp from web
            msq += delta * (z - mean)

        self.high_loss_train_FIM = mean
        self.high_loss_train_FIM_var = msq/(data_count-1)
        
    @tf.function
    def Get_Z(model,img):
        #returns the z value for a given x and y
        with tf.GradientTape() as tape:
            output = model(img,training=False)
            #sample from the output distribution
            output = tf.math.log(output[0,tf.random.categorical(tf.math.log(output), 1)[0][0]])

        grad = tape.gradient(output,model.trainable_variables) #all grads 
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
    def get_batch_loss(self,imgs,labels):
        preds = self.model(imgs,training=True)
        loss = self.loss_func(labels,preds)
        return loss

    @tf.function
    def test_step(self,imgs,labels):
        preds = self.model(imgs,training=False)
        loss = self.loss_func(labels,preds)
        self.test_loss_metric(loss)
        self.test_acc_metric(labels,preds)
        self.test_prec_metric(labels,preds)
        self.test_rec_metric(labels,preds)