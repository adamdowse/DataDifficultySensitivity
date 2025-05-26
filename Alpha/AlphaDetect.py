import tensorflow as tf
import numpy as np
import wandb
import os
import matplotlib.pyplot as plt

import AlphaData as AlphaData
import AlphaModels as AlphaModels
import AlphaUtils as AlphaUtils
import AlphaOptimizers as AlphaOptimizers





def compare_alpha(data_config, model_config):
    #load the model
    base_model = AlphaModels.load_model("Models/base_model.keras")
    #load the data
    train_dataset, test_dataset = AlphaData.load_data(data_config["train_test"])
    _, inv_test_dataset = AlphaData.load_data(data_config["inv_cols"])
    mnist_train_dataset, _ = AlphaData.load_data(data_config["mnist"])
    filled_train_dataset, _ = AlphaData.load_data(data_config["filled"])

    fig,ax = plt.subplots(1,1,figsize=(10, 10))
    limit = 5000
    t = "flat"
    train_alpha, train_y, train_o,train_imgs = AlphaUtils.calc_alpha_percentiles(train_dataset,
                                        base_model, 
                                        alpha_type=t,
                                        batch_size=None,
                                        limit=limit,
                                        ax=ax,
                                        ds_name="train")
    test_alpha, test_y, test_o,test_imgs = AlphaUtils.calc_alpha_percentiles(test_dataset, 
                                        base_model, 
                                        alpha_type=t,
                                        batch_size=None,
                                        limit=limit,
                                        ax=ax,  
                                        ds_name="test",
                                        og_alpha=train_alpha)
    inv_test_alpha, inv_test_y, inv_test_o,inv_test_imgs = AlphaUtils.calc_alpha_percentiles(inv_test_dataset, 
                                        base_model, 
                                        alpha_type=t,
                                        batch_size=None,
                                        limit=limit,
                                        ax=ax,
                                        ds_name="inv_test",
                                        og_alpha=train_alpha)
    mnist_alpha, mnist_y, mnist_o,mnist_imgs = AlphaUtils.calc_alpha_percentiles(mnist_train_dataset,
                                        base_model, 
                                        alpha_type=t,
                                        batch_size=None,
                                        limit=limit,
                                        ax=ax,
                                        ds_name="mnist",
                                        og_alpha=train_alpha)
    filled_alpha, filled_y, filled_o,filled_imgs = AlphaUtils.calc_alpha_percentiles(filled_train_dataset,
                                        base_model, 
                                        alpha_type=t,
                                        batch_size=None,
                                        limit=limit,
                                        ax=ax,
                                        ds_name="filled",
                                        og_alpha=train_alpha)

    #plot the imgs
    plt.title("Alpha Percentiles Comparison")
    plt.xlabel("-log(y_hat)")
    plt.ylabel("FIM_s")
    plt.legend()
    plt.xscale('log')
    plt.grid(True)
    plt.yscale("log")
    plt.savefig("alpha_comparison.png")

    AlphaUtils.plot_alpha_imgs(info={
        "train": train_imgs,
        "inv_test": inv_test_imgs,
        "test": test_imgs,
        "mnist": mnist_imgs,
        "filled": filled_imgs
    })

    #compare the alphas
    # wandb.log({
    #     "train_alpha": train_alpha,
    #     "inv_train_alpha": inv_train_alpha,
    #     "test_alpha": test_alpha,
    #     "inv_test_alpha": inv_test_alpha
    # })
    # print("Train alpha: ", train_alpha, train_y, train_o)
    # print("Inv Train alpha: ", inv_train_alpha, inv_train_y, inv_train_o)
    # print("Test alpha: ", test_alpha, test_y, test_o)
    # print("Inv Test alpha: ", inv_test_alpha, inv_test_y, inv_test_o)
    return


def main(data_config,model_config):

    train_dataset, test_dataset = AlphaData.load_data(data_config["train_test"])
    model = AlphaModels.create_model(model_config)
    model = AlphaModels.compile_model(model, model_config)
    
    #show the model summary
    print(model.summary())

    #show the example of a batch of data
    for item in train_dataset.take(1):
        x,y = item
        print("x shape: ", x.shape)
        print("y shape: ", y.shape)
        break

    #Train the model on the training data 
    #save the model
    model.fit(train_dataset, epochs=25, validation_data=test_dataset)
    model.save("Models/base_model.keras")






if __name__ == "__main__":
    # Initialize wandb
    #TODO FIXME split the data configs for train and test so that they can be different
    
    data_config = {
        "train_test": {
            'dataset_name': 'cifar10',
            'train_split': 0.8,
            'splits': ["train", "test"],
            'normalisation_method': 'div255',
            'y_mapping_type': ['onehot'],
            'filtering': None,
            "augmentations": None,
            'batch_size': 32,
            'shuffle': True,
            'buffer_size': 1000,
            'prefetch': True},
        "inv_cols": {
            'dataset_name': 'cifar10',
            'train_split': 0.8,
            'splits': ["train", "test"],
            'normalisation_method': 'div255',
            'augmentations': [{"name": "color_inv", "prob": 1}],
            'y_mapping_type': ['onehot'],
            'filtering': None,
            'batch_size': 32,
            'shuffle': False,
            'buffer_size': 1000,
            'prefetch': True},
        "mnist": {
            'dataset_name': 'mnist',
            'normalisation_method': 'div255',
            'splits': ["train", "test"],
            'train_split': 0.5,
            'y_mapping_type': ['onehot'],
            "augmentations": [{"name": "scale", "height": 32, "width": 32},
                             {"name": "to_color"}],
            'filtering': None,
            'batch_size': 32,
            'shuffle': False,
            'buffer_size': 1000,
            'prefetch': True},
        "filled": {
            'dataset_name': 'mnist',
            'normalisation_method': 'div255',
            'splits': ["train", "test"],
            'train_split': 0.5,
            'y_mapping_type': ['onehot'],
            "augmentations": [{"name": "scale", "height": 32, "width": 32},
                             {"name": "fill"},
                             {"name": "to_color"}],
            'filtering': None,
            'batch_size': 32,
            'shuffle': False,
            'buffer_size': 1000,
            'prefetch': True}
    }


    model_config = {
        'model_name': 'simple_cnn',
        'input_shape': (32,32,3),
        'layers': [
            {'type': 'conv2d', 'filters': 32, 'kernel_size': (3,3), 'activation': 'relu'},
            {'type': 'maxpooling2d', 'pool_size': (2,2)},
            {'type': 'conv2d', 'filters': 64, 'kernel_size': (3,3), 'activation': 'relu'},
            {'type': 'maxpooling2d', 'pool_size': (2,2)},
            {'type': 'conv2d', 'filters': 64, 'kernel_size': (3,3), 'activation': 'relu'},
            {'type': 'flatten'},
            {'type': 'dense', 'units': 64, 'activation': 'relu'},
            {'type': 'dense', 'units': 10, 'activation': 'softmax'}
        ],
        'optimizer': 'sgd',
        'loss': 'categorical_crossentropy',
        'metrics': ['accuracy'],
        'learning_rate': 0.001
    }
    os.environ['WANDB_API_KEY'] = 'fc2ea89618ca0e1b85a71faee35950a78dd59744'
    wandb.login()
    wandb.init(project="NewAlphaDetect", name="AlphaDetect", config= model_config | data_config)
    #main(data_config,model_config)
    compare_alpha(data_config, model_config)
    wandb.finish()