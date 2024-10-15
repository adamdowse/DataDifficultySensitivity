

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import wandb
import os
import argparse

import Mk2_Data as DataClass
import Mk3_Models as customModels
import Mk2_Funcs as FC

#main run file
def main(config):
    strategy = None
    #strategy = tf.distribute.MirroredStrategy()
    if strategy is not None:
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        config['batch_size'] = config['batch_size']*strategy.num_replicas_in_sync
        print('Batch size per replica: ',config['batch_size'])

    #load data
    print('Building Data')
    data = DataClass.Data(config)
    data.build_data()
    config.update({'steps_per_epoch':data.steps_per_epoch})

    #build model
    print('Building Model')
    if strategy is not None:
        with strategy.scope():
            model,callbacks = customModels.build_model(config)
    else:
        model,callbacks = customModels.build_model(config)

    #Defining Callbacks
    EOECallback = FC.CustomEOE(data.train_data,model,config,tf.keras.losses.CategoricalCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE))
    callbacks.append(EOECallback)
    if config['batch_calc_epoch_limit'] is not None and config['batch_calc_epoch_limit'] > 0:
        EOBCallback = FC.CustomEOB(data.train_data,model,config,tf.keras.losses.CategoricalCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE))
        callbacks.append(EOBCallback)
    #EOBCallback = FC.CustomEOB(data.train_data,model,config,tf.keras.losses.CategoricalCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE))
    wandbcallback = wandb.keras.WandbCallback(save_model=False)
    callbacks.append(wandbcallback)
    print('Callbacks:',callbacks)

    #show model summary
    print(model.model.summary(expand_nested=True))
    
    #train model
    model.fit(data.train_data,
        batch_size=config['batch_size'],
        shuffle=True,
        validation_data=data.test_data,
        epochs=config['epochs'],
        callbacks=callbacks)

if __name__ == '__main__':
    os.environ['WANDB_API_KEY'] = 'fc2ea89618ca0e1b85a71faee35950a78dd59744'
    wandb.login()
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-m', type=int, default='8', help='size of the maximisation set')
    parser.add_argument('-r', type=float, default='0.15', help='radius of the ball')
    parser.add_argument('-o', type=str, default='SGD', help='optimizer type')
    args = parser.parse_args()

    print('m: ',args.m)
    print('r: ',args.r)
    print('o: ',args.o)

    config = {'group':'fixednorm',
                'loss_func':'categorical_crossentropy',
                'data_name':'cifar10',
                'data_split':[0.9,0.1,0],
                'acc_sample_weight':None,
                'optimizer':args.o,
                'momentum':0,
                'dropout':0.0,
                'lr':0.001,
                'lr_decay_params': {'alpha':0.00001,'decay_steps':150*391},
                'lr_decay_type':'fixed', #fixed, exp_decay, percentage_step_decay,cosine_decay
                'batch_size':128,
                'label_smoothing':None,
                'model_init_type':None,
                'model_name':'CNN',
                'model_vars': None, #var = [max_features,sequence_length,embedding_dim]
                'num_classes':10,
                'img_size':(32,32,3),
                'augs': {"normalise":'resnet50','flip':'horizontal','crop':4}, #{'flip':'horizontal','crop':4,"normalise":'resnet50'},#{'flip':horizonatal,"crop":padding},
                'weight_reg':0.0005,
                'epochs': 200,

                'rho':args.r, # radius of ball 
                'rho_decay':1, # 1 = no decay
                'm':args.m, # must be less than batch size

                'norm_scale':1, #number fixes the norm of the gradient, 'sgd' uses the norm of the sgd grad, 'norm' uses the norm of the norm grad, 'boxcox' uses the boxcox adjusted norm
                'direction':'norm', #sgd, norm, boxcox
                
                'batch_calc_epoch_limit':0, #limit for using batch calcs and logging, if None or 0 then recording is off
                'batch_calc_freq':0,
                'epoch_calc_freq':1,
                'FIM_calc':False,
                'FIM_bs':5,
                'FIM_limit':1000,
                'FIM_groups':8,
                'Loss_spec_calc':False,
                'Grad_alignment_calc':False,
                'Grad_bs':5,
                }
    config['norm_name'] = 'dir_'+str(config['direction'])+'_scale_'+str(config['norm_scale'])
    wandb.init(project="NormSGD",config=config)
    main(config)