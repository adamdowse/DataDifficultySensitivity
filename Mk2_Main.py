

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import wandb
import os

import Mk2_Data as DataClass
import Mk2_Models as ModelClass
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
    F = FC.calc_FIM(data,model,FIM_bs,limit=limit,model_output_type='binary_logit')
    print('F matrix trace: ',F)
    wandb.log({'F_matrix_trace':F},step=epoch)

    # #Residuals matrix
    # R = FC.calc_residuals(data,model,limit=limit)
    # print('Residuals: ',R)
    # wandb.log({'Residuals':R},step=epoch)






#main run file
def main():
    #build model
    config = {'loss_func':'binary_crossentropy',
                'data_name':'imdb_reviews',
                'acc_sample_weight':None,
                'optimizer':'Adam',
                'lr':0.0001,
                'lr_decay_type':'fixed',
                'label_smoothing':None,
                'model_init_type':None,
                'model_name':'imdbConv1D',
                'model_vars': [10000,250,16], #var = [max_features,sequence_length,embedding_dim]
                'num_classes':2,
                'img_size':None,
                }
    strategy = None

    #load data
    data = DataClass.Data(config['data_name'],10,split=[0.8,0.2,0])
    data.build_data_in_mem()

    print('Building Model')
    model = ModelClass.Models(config,strategy)
    print(model.model.summary())
    print('Compiling Model')
    metric_limit = 1000
    data.build_train_iter(bs=1)
    compute_metrics(data,model,epoch=0,FIM_bs=5,limit=metric_limit)

    #train model
    
    print(model.model.summary())
    #print(data.get_batch())
    for i in range(15):
        #TODO There is a memory leak most likely with dataset building each epoch
        #tf error in converting index sliced
        print('Training Epoch: ',(i+1)*4)
        data.build_train_iter(bs=32)
        data.build_test_iter(bs=32)
        model.model.fit(data.train_data,validation_data=data.test_data,epochs=4,callbacks=[wandb.keras.WandbCallback(save_model=False)])
        data.build_train_iter(bs=1)
        compute_metrics(data,model,epoch=(i+1)*4,FIM_bs=5,limit=metric_limit)
    




if __name__ == '__main__':
    os.environ['WANDB_API_KEY'] = 'fc2ea89618ca0e1b85a71faee35950a78dd59744'
    wandb.login()
    wandb.init(project="DomainFIMs")
    main()