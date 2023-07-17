#This is the main file and should be used to run the project

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import Models
import DataHandler
import wandb
from tensorflow import keras
from wandb.keras import WandbCallback
import time
import tracemalloc
import os





def Main(config):
    print ("Main Started")
    
    #setup
    tf.keras.backend.clear_session()
    wandb.init(project='DataDiffSensTEST',config=config.__dict__)
    dataset = DataHandler.DataHandler(config)
    model = Models.Models(config,dataset.train_info)

    #Training
    while model.epoch_num <= config.epochs and not model.early_stop():
        print("Epoch: ",model.epoch_num,"Batch: ",model.batch_num)

        #data setup
        print("Data Setup")
        t = time.time()
        #need to test if we apply the method this epoch or not
        dataset.epoch_init(model,apply_method=True)
        model.epoch_init()
        print("Data and model Setup Time: ",time.time()-t)

        #Training
        print("Training")
        t = time.time()
        while dataset.current_train_data_points < dataset.total_train_data_points:
            imgs,labels = dataset.get_next()
            model.train_step(imgs,labels)
            model.batch_num += 1
        print("Epoch ",model.epoch_num, "Training Time: ",time.time()-t)

        #Testing
        print("Testing")
        t = time.time()
        model.model.evaluate(dataset.test_tfds)
        print("Testing Time: ",time.time()-t)

        #Record FIM
        if config.record_FIM:
            dataset.build_dataset(0)
            model.calc_FIM(dataset)
        
        if config.record_complex_FIM:
            model.calc_complex_FIM(dataset)
            
        #WandB logging
        model.log_metrics()

        #update counters
        model.epoch_num += 1
        dataset.epoch_num += 1

    
    tf.keras.backend.clear_session()
    print('Finished')





if __name__ == "__main__":
    #Config can be defined here
    class config_class:
    #/vol/research/NOBACKUP/CVSSP/scratch_4weeks/ad00878/datasets/
    #/com.docker.devenvironments.code/datasets/
        def __init__(self):
            self.batch_size = 100
            self.epochs = 10
            self.lr = 0.01
            self.lr_decay = 0
            self.lr_decay_type = 'fixed'
            self.lr_decay_end = 1000
            self.optimizer = 'SGD'
            self.loss_func = 'categorical_crossentropy'
            self.momentum = 0
            self.label_smoothing = 0
            self.seed = 1
            self.save_model = False
            self.weight_decay = 0
            self.data_augmentation = False
            self.data_augmentation_type = 'random'
            self.start_method_epoch = 1
            self.end_method_epoch = 2
            self.method = 'HighLossPercentage'
            self.method_param = 0.5
            self.record_FIM = True
            self.record_FIM_n_data_points = 1000
            self.record_complex_FIM = False
            self.data = 'MNIST'
            self.data_percentage = 1
            self.model_name = 'CNN'
            self.ds_path = '/com.docker.devenvironments.code/datasets/'
            self.group = 'test'
            self.early_stop = 10
        

    config = config_class()
    os.environ['WANDB_API_KEY'] = 'fc2ea89618ca0e1b85a71faee35950a78dd59744'
    wandb.login()
    Main(config)