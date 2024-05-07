

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import wandb
import os

import Mk2_Data as DataClass
import Mk2_Models as ModelClass
import Mk2_Funcs as FC



def compute_metrics(data,model,epoch,FIM_bs=5,limit=None):

    model.remove_softmax()
    G = FC.calc_G(data,model,limit=limit)
    #D = FC.calc_d2zdw2(data,model,limit=limit)
    model.add_softmax()


    #Compute the metrics
    #loss spectrum
    loss_spectrum = FC.calc_train_loss_spectrum(data,model,limit=limit)
    wandb.log({'loss_spectrum':loss_spectrum},step=epoch)

    #S matrix
    S = FC.calc_S(data,model,limit=limit)
    print('S matrix trace: ',S)
    wandb.log({'S_matrix_trace':S},step=epoch)

    #F matrix
    F = FC.calc_FIM(data,model,FIM_bs,limit=limit)
    print('F matrix trace: ',F)
    wandb.log({'F_matrix_trace':F},step=epoch)

    #Residuals matrix
    R = FC.calc_residuals(data,model,limit=limit)
    print('Residuals: ',R)
    wandb.log({'Residuals':R},step=epoch)

    #d2zdw2 matrix
    d2zdw2 = FC.calc_d2zdw2(data,model,limit=limit)





#main run file
def main():
    #load data
    data = DataClass.Data('mnist',10,split=[0.8,0.2,0])
    data.build_data_in_mem()

    #build model
    config = {'loss_func':'categorical_crossentropy',
                'acc_sample_weight':None,
                'optimizer':'SGD',
                'lr':0.01,
                'lr_decay_type':'fixed',
                'label_smoothing':None,
                'model_init_type':None,
                'model_name':'CNN',
                'num_classes':10,
                'img_size':(28,28,1),
                }
    strategy = None
    model = ModelClass.Models(config,strategy)

    data.build_train_iter(bs=1)
    compute_metrics(data,model,epoch=0,FIM_bs=5,limit=1000)
    




if __name__ == '__main__':
    os.environ['WANDB_API_KEY'] = 'fc2ea89618ca0e1b85a71faee35950a78dd59744'
    wandb.login()
    wandb.init(project="Hessian_Decomp_Test")
    main()