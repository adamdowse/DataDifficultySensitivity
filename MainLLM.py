#Main python file for the LLM project to test if it produces FIM phases

import numpy as np
import tensorflow as tf
import wandb
import time

import LLM #Where the LLM models are based
import LLMData #Where the data is loaded from
#import LLMUtils #Where the utility functions are


def Main(config):
    #Setup wandb
    wandb.init(project="LLM_FIM", config=config.__dict__)

    #Load the data
    data = LLMData.data_setup(config) #return the data classes

    #Setup the model
    print(config)
    print(data.train_ds)
    model = LLM.Model(config,data.train_ds) #Create the model

    text_batch, label_batch = data.get_train_batch()
    print(text_batch.shape)
    print(label_batch.shape)
    first_review, first_label = text_batch[0], label_batch[0]
    print("Review", first_review)
    print("Label", first_label)
    print("Vectorized review", model.__vectorize_text(first_review, first_label))




if __name__ == "__main__":
    #Setup the config
    class config_class:
    #/vol/research/NOBACKUP/CVSSP/scratch_4weeks/ad00878/datasets/
    #/com.docker.devenvironments.code/datasets/
        def __init__(self,args=None):
            self.vocab_size = 10000
            self.max_length = 256
            self.batch_size = 32
            self.dataset = "IMDB_reviews"
            

    #Run the main function
    config = config_class()
    Main(config)
