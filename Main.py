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
import argparse





def Main(config):
    print ("Main Started")
    
    #setup
    tf.keras.backend.clear_session()
    wandb.init(project='DataDiffSens',config=config.__dict__)
    dataset = DataHandler.DataHandler(config)
    model = Models.Models(config,dataset.train_info)

    #Training
    while model.epoch_num <= config.epochs and not model.early_stop():
        print("Epoch: ",model.epoch_num,"Batch: ",model.batch_num)

        #data setup
        print("Data Setup")
        t = time.time()
        #need to test if we apply the method this epoch or not
        #method_index is [(start_epoch,method),(start_epoch,method),...]
        if config.method_index is not None:
            if np.where(np.array(config.method_index)[:,0] == str(model.epoch_num_adjusted))[0].size > 0:
                method = config.method_index[np.where(np.array(config.method_index)[:,0] == str(model.epoch_num_adjusted))[0][0]][1]
                if method == 'Vanilla':
                    update = False
                elif method == 'HighLossPercentage':
                    update = True
        else:
            method = 'Vanilla'
            update = False

        dataset.epoch_init(model, method=method, update=update)
        model.epoch_init()
        print("Data and model Setup Time: ",time.time()-t)

        #Training
        print("Training")
        t = time.time()
        for i in range(dataset.num_batches):
            imgs,labels = dataset.__getitem__(i,training=True,return_loss=False)
            model.train_step(imgs,labels)
            model.batch_num += 1
        print("Epoch ",model.epoch_num, "Training Time: ",time.time()-t)

        #Testing
        print("Testing")
        t = time.time()
        model.test_results = model.model.evaluate(dataset.test_tfds)
        print("Testing Time: ",time.time()-t)

        #Record FIM
        if config.record_FIM:
            dataset.update_indexes_with_method(1,model,update=True,method='Vanilla')
            FullFIM, FullFIMVar = model.calc_FIM(dataset)
        
        if config.record_highloss_FIM:
            dataset.update_indexes_with_method(1,model,method='HighLossPercentage')
            HLFIM, HLFIMVar = model.calc_FIM(dataset)

        if config.record_lowloss_FIM:
            dataset.update_indexes_with_method(1,model,method='LowLossPercentage')
            LLFIM, LLFIMVar = model.calc_FIM(dataset)
        
        if config.record_staged_FIM:
            k = 8
            for i in range(k):
                dataset.update_indexes_with_method(1,model,method='Staged',update=False,stage=i,num_stages=k)
                staged_FIM, staged_FIMVar = model.calc_FIM(dataset)
            
                wandb.log({'StagedFIM_'+str(i):staged_FIM,'StagedFIMVar_'+str(i):staged_FIMVar},step=model.epoch_num_adjusted)

        #Record Loss Spectrum
        if config.record_loss_spectrum:
            dataset.update_indexes_with_method(1,model,update=True,method='Vanilla')
            loss_spectrum = model.calc_loss_spectrum(dataset)
            wandb.log({'LossSpectrum':loss_spectrum},step=model.epoch_num_adjusted)
            
        #WandB logging
        model.log_metrics()
        if config.record_FIM:
            wandb.log({'FullFIM':FullFIM,'FullFIMVar':FullFIMVar},step=model.epoch_num_adjusted)
        if config.record_highloss_FIM:
            wandb.log({'HLFIM':HLFIM,'HLFIMVar':HLFIMVar},step=model.epoch_num_adjusted)
        if config.record_lowloss_FIM:
            wandb.log({'LLFIM':LLFIM,'LLFIMVar':LLFIMVar},step=model.epoch_num_adjusted)

        #update counters
        #if the method is being applied then epoch is updated with the percentage used
        if method == 'HighLossPercentage':
            model.epoch_num_adjusted += config.method_param
        else:
            model.epoch_num_adjusted += 1
        model.epoch_num += 1
        dataset.epoch_num += 1

    
    tf.keras.backend.clear_session()
    print('Finished')





if __name__ == "__main__":
    #Config can be defined here
    #argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--method_index',type=str,default="0 Vanilla")
    parser.add_argument('--percent',type=float,default=0.5)
    class config_class:
    #/vol/research/NOBACKUP/CVSSP/scratch_4weeks/ad00878/datasets/
    #/com.docker.devenvironments.code/datasets/
        def __init__(self,args=None):
            self.batch_size = 128
            self.epochs = 200
            self.lr = 0.01 #0.001 is adam preset in tf
            self.lr_decay_type = 'fixed'
            self.lr_decay_param = []
            self.optimizer = 'SGD'
            self.loss_func = 'categorical_crossentropy'
            self.momentum = 0
            self.label_smoothing = 0
            self.seed = 1
            self.save_model = False
            self.weight_decay = 0
            self.data_augmentation = False
            self.data_augmentation_type = None
            args.method_index = args.method_index.split(' ')
            self.method_index = [[int(args.method_index[i][0]),args.method_index[i+1]] for i in range(0,len(args.method_index),2)]
            self.method_index = [(i[0],str(i[1])) for i in self.method_index]
            self.method_param = args.percent
            self.record_FIM = True
            self.record_highloss_FIM = True
            self.record_lowloss_FIM = True
            self.record_staged_FIM = False
            self.record_FIM_n_data_points = 5000
            self.record_loss_spectrum = False
            self.data = 'cifar10'
            self.data_percentage = 1
            self.model_name = 'ResNet18' #CNN, ResNet18, ACLCNN,ResNetV1-14
            self.model_init_type = None
            self.model_init_seed = np.random.randint(0,100000)
            self.ds_path = '/com.docker.devenvironments.code/datasets/'
            self.group = 'T4_Staged_FIM'
            self.early_stop = 20
            self.early_stop_epoch = 40
        

    config = config_class(args=parser.parse_args())
    os.environ['WANDB_API_KEY'] = 'fc2ea89618ca0e1b85a71faee35950a78dd59744'
    wandb.login()
    Main(config)