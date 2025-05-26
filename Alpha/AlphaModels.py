
import tensorflow as tf
import numpy as np


def create_model(model_config,custom_training_func=None,custom_test_func=None):
    model_name = model_config['model_name']

    if model_name == "simple_cnn":
        model = SimpleCNNModel(model_config)
    
    else:
        raise ValueError(f"Unknown model name: {model_name}.")
    
    #custom training loop
    if custom_training_func is not None:
        model.train_step = custom_training_func
    else:
        print("Using default training loop.")
    #custom test loop
    if custom_test_func is not None:
        model.test_step = custom_test_func
    else:
        print("Using default test loop.")
    
    return model
    
def load_model(model_path):
    if not tf.io.gfile.exists(model_path):
        raise ValueError(f"Model path {model_path} does not exist.")
    
    model = tf.keras.models.load_model(model_path, custom_objects={'SimpleCNNModel': SimpleCNNModel})
    return model


def compile_model(model, model_config):
    optimizer = model_config['optimizer']
    loss = model_config['loss']
    metrics = model_config['metrics']

    #TODO add support for custom loss functions, custom metrics, and custom optimizers, and custom callbacks

    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=model_config['learning_rate'])
    elif optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=model_config['learning_rate'])
    elif optimizer == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=model_config['learning_rate'])
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}. Supported optimizers are: adam, sgd, rmsprop.")

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    #build the model
    if model_config['input_shape'] is not None:
        model.build(input_shape=model_config['input_shape'])
    else:
        raise ValueError("Input shape must be specified in the model config.")
    return model

#----Models----

class SimpleCNNModel(tf.keras.Model):
    def __init__(self, model_config):
        super(SimpleCNNModel, self).__init__()
        self.model_config = model_config
        self.model_layers = []
        if model_config['input_shape'] is None:
            raise ValueError("Input shape must be specified in the model config.")
        if model_config['layers'] is None:
            raise ValueError("Layers must be specified in the model config.")
        

        for i,layer_info in enumerate(model_config['layers']):
            if layer_info["type"] == "conv2d":
                if i == 0:
                    self.model_layers.append(tf.keras.layers.Conv2D(layer_info['filters'], layer_info["kernel_size"], activation=layer_info['activation'], input_shape=model_config['input_shape']))
                else:
                    self.model_layers.append(tf.keras.layers.Conv2D(layer_info['filters'], layer_info["kernel_size"], activation=layer_info['activation']))
            if layer_info['type'] == "maxpooling2d":
                self.model_layers.append(tf.keras.layers.MaxPooling2D(layer_info['pool_size']))
            if layer_info['type'] == "flatten":
                self.model_layers.append(tf.keras.layers.Flatten())
            if layer_info['type'] == "dense":
                self.model_layers.append(tf.keras.layers.Dense(layer_info['units'], activation=layer_info['activation']))

    @tf.function
    def call(self, x):
        for layer in self.model_layers:
            x = layer(x)
        return x
    
    @classmethod
    def from_config(cls, config):
        return cls(config["model_config"])
    
    def get_config(self):
        return {"model_config": self.model_config}

    def build(self, input_shape):
        input_shape = tuple(input_shape)
        if len(input_shape) == 3:
            zeros_shape = (1,) + input_shape
        elif input_shape[0] is None:
            zeros_shape = (1,) + input_shape[1:]
        else:
            zeros_shape = input_shape
        x = tf.zeros(zeros_shape)
        for layer in self.model_layers:
            x = layer(x)
        super().build(input_shape)
    
    def get_build_config(self):
        # Return the config needed to build the model (usually input_shape)
        return {"input_shape": self.model_config["input_shape"]}

    def build_from_config(self, config):
        # Actually build the model from the config
        self.build(config["input_shape"])
    



