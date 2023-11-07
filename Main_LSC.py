#This is the main file and should be used to run the project

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import Models
import DataHandlerV2 as DataHandler
import wandb
from tensorflow import keras
from wandb.keras import WandbCallback
import time
import tracemalloc
import os
import argparse


#This run file is used for the large scale curvature and min fim data count experiments

def Main(config):
    print ("Main Started")
    
    #setup
    tf.keras.backend.clear_session()
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    wandb.init(project='DataDiffSens',config=config.__dict__)
    #dataset = DataHandler.DataHandler(config)
    data_obj = DataHandler.Data(strategy,config.ds_root,config.preaugment,config.img_size)
    data_obj_2 = DataHandler.Data(strategy,config.ds_root,config.preaugment,config.img_size)
    test_data_obj = DataHandler.Data(strategy,config.ds_root,config.preaugment,config.img_size)
    test_ds, test_num_batches = test_data_obj.init_data(config.batch_size,train=False,distributed=True,shuffle=False)
    with strategy.scope():
        model = Models.Models(config,data_obj.num_classes,strategy)
    
    FIM_BS = 10

    #FIM before training
    #data_obj_2.reduce_data(method='all')
    #ds2,num_batches2 = data_obj_2.init_data(FIM_BS,train=True,distributed=True,shuffle=True)
    #OGFIM = model.calc_dist_FIM(ds2,num_batches2,FIM_BS)
    #wandb.log({'OriginalFIM':OGFIM},step=0)
    
    

    #Training
    epoch_num = 0 #this is the epoch number
    adjusted_epoch_num = 0 #this is the epoch number acounting for percentage epochs
    while epoch_num <= config.epochs and not model.early_stop(adjusted_epoch_num):
        print("Epoch: ",model.epoch_num,"Batch: ",model.batch_num)

        #data setup
        print("Data Setup")
        t = time.time()
        config.method_param = "all"
        print(config.method_param)
        data_obj.reduce_data("all",[config.method_param])
        ds, num_batches = data_obj.init_data(config.batch_size,train=True,distributed=False,shuffle=True)
        model.epoch_init()
        print("Data and model Setup Time: ",time.time()-t)


        #Training
        print("Training")
        t = time.time()
        total_loss = 0
        batch_count = 0
        ds_iter = iter(ds)
        print("Number of batches: ",num_batches)
        for _ in range(num_batches):
            if batch_count%100 == 0:
                print("Batch: ",batch_count)
            total_loss += model.distributed_train_step(next(ds_iter))
            batch_count += 1
        train_loss = total_loss/num_batches
        print("Training Time: ",time.time()-t)

        #Testing
        print("Testing")
        t = time.time()
        test_ds_iter = iter(test_ds)
        print("Number of batches: ",test_num_batches)
        batch_count = 0
        total_loss = 0
        for _ in range(test_num_batches):
            if batch_count%50 == 0:
                print("Batch: ",batch_count)
            total_loss += model.distributed_test_step(next(test_ds_iter))
            batch_count += 1
        test_loss = total_loss/test_num_batches
        print("Testing Time: ",time.time()-t)

        #Record FIM
        for i in [40000,10000,2500,500,100]:
            for j in range(10):
                print("Recording FIM: ",i," ",j)
                data_obj.reduce_data(method='all')
                ds,num_batches = data_obj.init_data(FIM_BS,train=True,distributed=True,shuffle=True)
                FullFIM = model.calc_dist_FIM(ds,num_batches,FIM_BS,record_FIM_n_data_points=i)
                wandb.log({'FullFIM_'+str(i)+'_'+str(j):FullFIM},step=epoch_num)
        
            
        #WandB logging
        model.log_metrics(train_loss,test_loss,epoch_num,adjusted_epoch_num)

        #update counters
        adjusted_epoch_num += 1
        epoch_num += 1


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
            self.batch_size = 100           #batch size
            self.lr = 0.001                #0.001 is adam preset in tf
            self.lr_decay_type = 'fixed'    #fixed, exp
            self.lr_decay_param = [1e-7]     #defult adam = [eplioon = 1e-7] SGD exp= [decay steps, decay rate]
            self.optimizer = 'Adam'         #Adam, SGD, RMSprop
            self.loss_func = 'categorical_crossentropy'
            self.momentum = 0               #momentum for SGD  

            #length of training
            self.epochs = 100             #max number of epochs
            self.early_stop = 150           #number of epochs below threshold before early stop
            self.early_stop_epoch = 150     #epoch to start early stop
            self.steps_per_epoch = 1000      #number of batches per epoch

            #Results
            self.group = 'T3_mindataFIM'
            self.acc_sample_weight = None #for HAM [1,1,1,1,5,1,1] for CIFAR [1,1,1,1,1,1,1,1,1,1]
            self.record_FIM = False                 #record the full FIM    
            self.record_highloss_FIM = False        #record the FIM of the high loss samples
            self.record_lowloss_FIM = False         #record the FIM of the low loss samples
            self.record_staged_FIM = False          #record the FIM of the staged loss samples
            self.record_FIM_n_data_points = 5000    #number of data points to use for FIM
            self.record_loss_spectrum = False       #record the loss spectrum
            self.record_original_FIM = False         #record the FIM before any training is done
            self.record_step_FIM = False             #record the FIM after each step
            
            #Data
            self.data = 'cifar10'          #cifar10 HAM10000 SVHN
            self.img_size = (32,32,3)       #size of images (299,299) for IRv2
            self.ds_root = '/com.docker.devenvironments.code/CIFAR10/' #root path of dataset
            self.data_percentage = 1        #1 is full dataset HAM not implemented
            self.preaugment = 0         #number of images to preaugment
            self.label_smoothing = 0        #0 is no smoothing
            self.misslabel = 0              #0 is no misslabel

            #Model
            self.model_name = 'TFCNN'    #CNN, ResNet18, ACLCNN,ResNetV1-14,TFCNN,IRv2(has ImageNet weights)
            self.model_init_type = None #Not recomended
            self.model_init_seed = np.random.randint(0,100000)
            self.weight_decay = 0      #0.0001 is default for adam

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
