#This is the main file and should be used to run the project

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import Models.py as Models
import DataHandler.py as DataHandler
import wandb
from tensorflow import keras
from wandb.keras import WandbCallback
import time
import tracemalloc
import os





def Main(config):
    #this is the main function
    print ("Main Started")
    
    #setup
    tf.keras.backend.clear_session()
    wandb.init(project='DataDiffSens',config=config.__dict__)
    dataset = DataHandler.DataHandler(config)
    model = Models.Models(config)

    #Training
    while model.epoch_num <= config.epochs and not model.early_stop():
        print("Epoch: ",model.epoch_num,"Batch: ",model.batch_num)

        #data setup
        print("Data Setup")
        t = time.time()
        dataset.epoch_init()
        model.epoch_init()
        print("Data and model Setup Time: ",time.time()-t)

        #Training
        print("Training")
        t = time.time()
        while dataset.current_train_batch_num < dataset.train_batches:
            imgs,labels = dataset.get_train_batch()
            model.train_step(imgs,labels)
            model.batch_num += 1
        print("Training Time: ",time.time()-t)

        #Testing
        print("Testing")
        t = time.time()
        while dataset.current_test_batch_num < dataset.test_batches:
            imgs,labels = dataset.get_test_batch()
            model.test_step(imgs,labels)
        print("Testing Time: ",time.time()-t)

        #Record FIM
        if config.record_FIM:
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
            self.batch_size = 64
            self.epochs = 10
            self.lr = 0.01
            self.lr_decay = 0.5
            self.lr_decay_type = 'step'
            self.lr_decay_end = 1000
            self.optimizer = 'SGD'
            self.momentum = 0
            self.seed = 1
            self.save_model = False
            self.weight_decay = 0
            self.data_augmentation = False
            self.data_augmentation_type = 'random'
            self.start_method_epoch = 0
            self.end_method_epoch = 0
            self.method = 'baseline'
            self.record_FIM = False
            self.record_FIM_n_data_points = 5000
            self.record_complex_FIM = False
            self.data = 'MNIST'
            self.model_name = 'CNN'
            self.ds_path = './data'
            self.group = 'default'
            self.early_stop = 10
        
        

    config = config_class()

    Main(config)