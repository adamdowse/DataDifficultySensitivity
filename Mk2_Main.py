

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

def compute_metrics(data,model,epoch,FIM_bs=1,limit=None):

    
    #model.make_softmax_model() This may be needed for current model?

    #print(model.count_params())
    #compute R
    #R = FC.calc_R(data,model,limit=limit)
    #wandb.log({'R_matrix_trace':R},step=epoch)

    #[G_mean,S_mean,dzdt2_mean] = FC.calc_G(data,model,limit=limit)
    #wandb.log({'G_matrix_trace':G_mean},step=epoch)
    #wandb.log({'S_matrix_trace':S_mean},step=epoch)
    #wandb.log({'dzdt2_matrix_trace':dzdt2_mean},step=epoch)

    #wandb.log({'FullH_matrix_trace':G_mean+R},step=epoch)
    #D = FC.calc_d2zdw2(data,model,limit=limit)

    #Compute the metrics
    #loss spectrum
    #loss_spectrum = FC.calc_train_loss_spectrum(data,model,limit=limit)
    #wandb.log({'loss_spectrum':loss_spectrum},step=epoch)

    # #S matrix
    # S = FC.calc_S(data,model,limit=limit)
    # print('S matrix trace: ',S)
    # wandb.log({'S_matrix_trace':S},step=epoch)

    # #F matrix
    F = FC.calc_FIM(data,model,FIM_bs,limit=limit,model_output_type='softmax')
    print('F matrix trace: ',F)
    wandb.log({'F_matrix_trace':F},step=epoch)

    # #Residuals matrix
    # R = FC.calc_residuals(data,model,limit=limit)
    # print('Residuals: ',R)
    # wandb.log({'Residuals':R},step=epoch)






#main run file
def main(config):
    #build model
    
    #strategy = None
    strategy = tf.distribute.MirroredStrategy()
    if strategy is not None:
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        config['batch_size'] = config['batch_size']*strategy.num_replicas_in_sync
        print('Batch size per replica: ',config['batch_size'])

    #load data
    print('Building Data')
    data = DataClass.Data(config)
    data.build_data()
    config.update({'steps_per_epoch':data.steps_per_epoch})

    print('Building Model')
    if strategy is not None:
        with strategy.scope():
            model,callbacks = customModels.build_model(config)
    else:
        model,callbacks = customModels.build_model(config)
    #model.model.summary()
    wandbcallback = wandb.keras.WandbCallback(save_model=False)
    callbacks.append(wandbcallback)
    print('Callbacks:',callbacks)
    
    metric_limit = 1000
    #compute_metrics(data,model,epoch=0,FIM_bs=5,limit=metric_limit)
    
    #train model
    # epochs_per_step = 1
    # for i in range(config['epochs']):
    #     #TODO There is a memory leak most likely with dataset building each epoch
    #     #tf error in converting index slices to tensors
    #     print('Training Epoch: ',(i+1)*epochs_per_step)
    #     model.fit(data.train_data,batch_size=config['batch_size'],shuffle=True,validation_data=data.test_data,epochs=epochs_per_step,callbacks=[wandb.keras.WandbCallback(save_model=False)])
    #     #compute_metrics(data,model,epoch=(i+1)*epochs_per_step,FIM_bs=5,limit=metric_limit)
    #lr_callback = tf.keras.callbacks.LearningRateScheduler(customModels.lr_selector(config['lr_decay_type'],config), verbose=1)
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

    config = {'group':'test',
                'loss_func':'categorical_crossentropy',
                'data_name':'cifar10',
                'data_split':[0.8,0.2,0],
                'acc_sample_weight':None,
                'optimizer':args.o,
                'momentum':0.9,
                'lr':0.1,
                'lr_decay_params': {'lr_decay_rate':0.1,'lr_decay_epochs_percent':[0.5,0.75]},
                'lr_decay_type':'percentage_step_decay', #fixed, exp_decay, percentage_step_decay
                'batch_size':128,
                'label_smoothing':None,
                'model_init_type':None,
                'model_name':'ResNet18V2',
                'model_vars': None, #var = [max_features,sequence_length,embedding_dim]
                'num_classes':10,
                'img_size':(32,32,3),
                'rho':args.r, # radius of ball 
                'rho_decay':1, # 1 = no decay
                'm':args.m, # must be less than batch size
                'augs': {'flip':'horizontal','crop':4},#{'flip':horizonatal,"crop":padding},
                'weight_reg':0.0,
                'epochs': 200,
                }
    wandb.init(project="SAM",config=config)
    main(config)