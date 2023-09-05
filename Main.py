#This is the main file and should be used to run the project

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import Models
import DataHandler
import CustomImgGen
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
    #wandb.init(project='DataDiffSens',config=config.__dict__)
    #dataset = DataHandler.DataHandler(config)
    dataset = CustomImgGen.CustomImageGen(
        config.data,
        config.ds_path,
        config.meta_data_path,
        config.batch_size,
        config.img_size,
        trainsize = config.train_test_split,
        model_name = config.model_name)
    model = Models.Models(config,dataset.num_classes)
    #model.config.weighted_train_acc_sample_weight = dataset.config.weighted_train_acc_sample_weight

    #Training
    while model.epoch_num <= config.epochs and not model.early_stop():
        print("Epoch: ",model.epoch_num,"Batch: ",model.batch_num)

        #data setup
        print("Data Setup")
        t = time.time()
        #need to test if we apply the method this epoch or not
        #method_index is [(start_epoch,method),(start_epoch,method),...] and start_epoch is a float
        if config.method_index is not None:
            print('Method Index',config.method_index)
            print('Adjusted Epoch',str(model.epoch_num_adjusted))
            if np.where(np.array(config.method_index)[:,0] == str(model.epoch_num_adjusted))[0].size > 0:
                method = config.method_index[np.where(np.array(config.method_index)[:,0] == str(model.epoch_num_adjusted))[0][0]][1]
                print('Updating Method Used to ',method)
                if method == 'Vanilla':
                    update = False
                elif method == 'HighLossPercentage':
                    update = True
                else:
                    print("ERROR:Method not recognised")
        else:
            method = 'Vanilla'

        #update the loss of all train data if needed
        if update and model.epoch_num == 0:
            dataset.update_losses(model)
        dataset.update_mask(method=method)
        dataset.build_batches()
        model.epoch_init()
        print("Data and model Setup Time: ",time.time()-t)

        #Training
        print("Training")
        t = time.time()
        for i in range(dataset.num_batches):
            imgs,labels = dataset.__getitem__(i)
            model.train_step(imgs,labels)
            model.batch_num += 1
        print("Epoch ",model.epoch_num, "Training Time: ",time.time()-t)

        dataset.on_epoch_end(method=method)

        #Testing
        print("Testing")
        t = time.time()
        for i in range(dataset.num_test_batches):
            imgs,labels = dataset.__getitem__(i,training_set=False)
            model.test_step(imgs,labels)
        #model.test_results = model.model.evaluate(dataset.test_tfds)
        print("Testing Time: ",time.time()-t)

        #does the next epoch need loss updates
        if config.method_index is not None:
            if np.where(np.array(config.method_index)[:,0] == str(model.epoch_num_adjusted+1))[0].size > 0:
                method = config.method_index[np.where(np.array(config.method_index)[:,0] == str(model.epoch_num_adjusted+1))[0][0]][1]
                if method == 'Vanilla':
                    update = False
                elif method == 'HighLossPercentage': #CAN ADD HERE FOR OTHER METHODS
                    update = True
            else:
                update = False
        else:
            update = False

        if config.record_FIM or config.record_highloss_FIM or config.record_lowloss_FIM or config.record_staged_FIM or config.record_loss_spectrum or update:
            #update the loss of all train data
            dataset.update_losses(model)

        #Record FIM
        if config.record_FIM:
            dataset.update_mask(method='All')
            dataset.build_batches(batch_size=1)
            FullFIM, FullFIMVar = model.calc_FIM(dataset)
        
        if config.record_highloss_FIM:
            dataset.update_mask(method='Loss',split_type='High',percentage=0.5)
            dataset.build_batches(batch_size=1)
            HLFIM, HLFIMVar = model.calc_FIM(dataset)

        if config.record_lowloss_FIM:
            dataset.update_mask(method='Loss',split_type='Low',percentage=0.5)
            dataset.build_batches(batch_size=1)
            LLFIM, LLFIMVar = model.calc_FIM(dataset)
        
        if config.record_staged_FIM:
            k = 8
            for i in range(k):
                dataset.update_mask(method='Loss',split_type='Staged',percentage=1/k,stage=i)
                dataset.build_batches(batch_size=1)
                staged_FIM, staged_FIMVar = model.calc_FIM(dataset)
            
                wandb.log({'StagedFIM_'+str(i):staged_FIM,'StagedFIMVar_'+str(i):staged_FIMVar},step=model.epoch_num)

        #Record Loss Spectrum
        if config.record_loss_spectrum:
            dataset.update_mask(method='All')
            dataset.build_batches(batch_size=1)
            loss_spectrum = model.calc_loss_spectrum(dataset)
            wandb.log({'LossSpectrum':loss_spectrum},step=model.epoch_num)
            
        #WandB logging
        model.log_metrics()
        if config.record_FIM:
            wandb.log({'FullFIM':FullFIM,'FullFIMVar':FullFIMVar},step=model.epoch_num)
        if config.record_highloss_FIM:
            wandb.log({'HLFIM':HLFIM,'HLFIMVar':HLFIMVar},step=model.epoch_num)
        if config.record_lowloss_FIM:
            wandb.log({'LLFIM':LLFIM,'LLFIMVar':LLFIMVar},step=model.epoch_num)

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
            #Hyperparameters
            self.batch_size = 16            #batch size
            self.lr = 0.01                  #0.001 is adam preset in tf
            self.lr_decay_type = 'fixed'    #fixed, exp
            self.lr_decay_param = [0.1]     #defult adam = [eplioon = 1e-7] SGD exp= [decay steps, decay rate]
            self.optimizer = 'Adam'         #Adam, SGD, RMSprop
            self.loss_func = 'categorical_crossentropy'
            self.momentum = 0               #momentum for SGD  

            #length of training
            self.epochs = 150               #max number of epochs
            self.early_stop = 150           #number of epochs below threshold before early stop
            self.early_stop_epoch = 150     #epoch to start early stop

            #Results
            self.group = 'TestT5'
            self.weighted_train_acc_sample_weight = [1,1,1,1,5,1,1] #for HAM [1,1,1,1,5,1,1] for CIFAR [1,1,1,1,1,1,1,1,1,1]
            self.record_FIM = False                 #record the full FIM    
            self.record_highloss_FIM = False        #record the FIM of the high loss samples
            self.record_lowloss_FIM = False         #record the FIM of the low loss samples
            self.record_staged_FIM = False          #record the FIM of the staged loss samples
            self.record_FIM_n_data_points = 5000    #number of data points to use for FIM
            self.record_loss_spectrum = False       #record the loss spectrum
            
            #Data
            self.data = 'HAM10000'          #cifar10 HAM10000
            self.img_size = (299,299,3)       #size of images (299,299) for IRv2
            self.ds_path = '/com.docker.devenvironments.code/HAM10000' #root path of dataset
            self.meta_data_path = '/com.docker.devenvironments.code/HAM10000/HAM10000_metadata.csv' #path to csv
            self.data_percentage = 1        #1 is full dataset HAM not implemented
            self.train_test_split = 0.85    #percentage of data to use for training
            self.label_smoothing = 0        #0 is no smoothing
            self.misslabel = 0              #0 is no misslabel
            self.data_augmentation = False  #Not implemented
            self.data_augmentation_type = None #Not implemented

            #Model
            self.save_model = False     #not implemented
            self.weight_decay = 0       #not fully implemented
            self.model_name = 'IRv2'    #CNN, ResNet18, ACLCNN,ResNetV1-14,TFCNN,IRv2(has ImageNet weights)
            self.model_init_type = None #Not recomended
            self.model_init_seed = np.random.randint(0,100000)

            #Method
            args.method_index = args.method_index.split(' ')    #inputed as 'start_epoch method start_epoch method ...'
            self.method_index = [[args.method_index[i],args.method_index[i+1]] for i in range(0,len(args.method_index),2)]
            self.method_index = [(float(i[0]),str(i[1])) for i in self.method_index] # [(start_epoch,method),(start_epoch,method),...]
            self.method_param = args.percent #percentage of data to use for method

            #Misc
            self.seed = 1
            
            
            

            
            
            
            
        

    config = config_class(args=parser.parse_args())
    os.environ['WANDB_API_KEY'] = 'fc2ea89618ca0e1b85a71faee35950a78dd59744'
    wandb.login()
    Main(config)
