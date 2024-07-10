#testing the generator on multi gpus
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
import time


import Mk3_Models as customModels

def model_test():
    config = {
        'model_name':'PA_ResNet18',
        'img_size':(32,32,3),
        'model_init_type': None,
        'num_classes':10,
        'batch_size':32,
        'epochs':10,
        'learning_rate':0.001,
        'loss':'categorical_crossentropy',
        'metrics':['accuracy']
    }
    model = customModels.build_model(config)






if __name__ == "__main__":
    #main()
    model_test()