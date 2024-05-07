

import tensorflow as tf
import wandb   
from tensorflow import keras
from keras import layers

import math
import time
import numpy as np

#File that holds all the custom loss functions as a keras loss class

class WeightedCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, class_weights, reduction=tf.keras.losses.Reduction.NONE, from_logits=False, label_smoothing=0, name=None):
        super().__init__(reduction, name)
        self.label_smoothing = label_smoothing
        self.from_logits = from_logits
        self.class_weights = class_weights # shape (num_classes,)
        self.class_weights = tf.cast(self.class_weights, tf.float32)
    
    #def __call__(self, y_true, y_pred, sample_weight=self.class_weights):
    #    return super().__call__(y_true, y_pred, sample_weight)

    def weighted_categorical_crossentropy(self,y_true, y_pred):
        # Calculate the loss between y_pred and y_true as batches
        if self.from_logits:
            y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
        # Calculate the loss between y_pred and y_true
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits= self.from_logits,label_smoothing=self.label_smoothing)
        # Make class_weights broadcastable and multiply them with the losses
        class_weights = tf.broadcast_to(self.class_weights, [tf.shape(loss)[0],tf.shape(self.class_weights)[0]])
        # Calculate the weight vector shape (batch_size,num_classes) * (batch_size, num_classes) = (batch_size, num_classes)
        weights = tf.reduce_sum(y_true * class_weights, axis=1)
        # Apply the weights to the loss
        loss = tf.multiply(loss, weights)
        
        #deal with reduction
        if self.reduction == tf.keras.losses.Reduction.SUM:
            return tf.reduce_sum(loss)
        elif self.reduction == tf.keras.losses.Reduction.NONE:
            return loss
        elif self.reduction == tf.keras.losses.Reduction.AUTO:
            return tf.reduce_mean(loss)
        else:
            print('Reduction not recognised')
        
    def call(self, y_true, y_pred):
        return self.weighted_categorical_crossentropy(y_true, y_pred)
