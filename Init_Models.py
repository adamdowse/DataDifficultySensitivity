import tensorflow as tf
import numpy as np


def CNN3(img_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=img_shape),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(64,3,activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(num_classes,activation='softmax')
    ])
    return model

def CNN4(img_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=img_shape),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(64,3,activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(64,3,activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(num_classes,activation='softmax')
    ])
    return model

def CNN5(img_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=img_shape),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(64,3,activation='relu'),
        tf.keras.layers.Conv2D(64,3,activation='relu'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(128,3,activation='relu'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(num_classes,activation='softmax')
    ])
    return model

def CNN5_NoPool(img_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=img_shape),
        tf.keras.layers.Conv2D(64,3,activation='relu'),
        tf.keras.layers.Conv2D(64,3,activation='relu'),
        tf.keras.layers.Conv2D(128,3,activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(num_classes,activation='softmax')
    ])
    return model

def CNN5_Dense1(img_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=img_shape),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(64,3,activation='relu'),
        tf.keras.layers.Conv2D(64,3,activation='relu'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(128,3,activation='relu'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(num_classes,activation='softmax')
    ])
    return model

def CNN5_Dense2(img_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=img_shape),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(64,3,activation='relu'),
        tf.keras.layers.Conv2D(64,3,activation='relu'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(128,3,activation='relu'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(num_classes,activation='softmax')
    ])
    return model

def CNN5_Dense3(img_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=img_shape),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(64,3,activation='relu'),
        tf.keras.layers.Conv2D(64,3,activation='relu'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(128,3,activation='relu'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(num_classes,activation='softmax')
    ])
    return model

def CNN5_Dense4(img_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=img_shape),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(64,3,activation='relu'),
        tf.keras.layers.Conv2D(64,3,activation='relu'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(128,3,activation='relu'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(num_classes,activation='softmax')
    ])
    return model

def CNN5_Dense5(img_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=img_shape),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(64,3,activation='relu'),
        tf.keras.layers.Conv2D(64,3,activation='relu'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(128,3,activation='relu'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(num_classes,activation='softmax')
    ])
    return model

def CNN5_DenseL(img_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=img_shape),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(64,3,activation='relu'),
        tf.keras.layers.Conv2D(64,3,activation='relu'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(128,3,activation='relu'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256,activation='relu'),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(num_classes,activation='softmax')
    ])
    return model

def CNN5_DenseXL(img_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=img_shape),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(64,3,activation='relu'),
        tf.keras.layers.Conv2D(64,3,activation='relu'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(128,3,activation='relu'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512,activation='relu'),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(num_classes,activation='softmax')
    ])
    return model

def CNN5_DenseXXL(img_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=img_shape),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(64,3,activation='relu'),
        tf.keras.layers.Conv2D(64,3,activation='relu'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(128,3,activation='relu'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024,activation='relu'),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(num_classes,activation='softmax')
    ])
    return model


def CNN6(img_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=img_shape),
        tf.keras.layers.Conv2D(32,3,activation='relu'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(64,3,activation='relu'),
        tf.keras.layers.Conv2D(64,3,activation='relu'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(128,3,activation='relu'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(num_classes,activation='softmax')
    ])
    return model

def CNN7(img_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=img_shape),
        tf.keras.layers.Conv2D(32,3,activation='relu'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(64,3,activation='relu'),
        tf.keras.layers.Conv2D(64,3,activation='relu'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(128,3,activation='relu'),
        tf.keras.layers.Conv2D(128,3,activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(num_classes,activation='softmax')
    ])
    return model

def CNN8(img_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=img_shape,padding='same'),
        tf.keras.layers.Conv2D(32,3,activation='relu',padding='same'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(64,3,activation='relu',padding='same'),
        tf.keras.layers.Conv2D(64,3,activation='relu',padding='same'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(128,3,activation='relu',padding='same'),
        tf.keras.layers.Conv2D(128,2,activation='relu',padding='same'),
        tf.keras.layers.Conv2D(256,3,activation='relu',padding='same'),
        tf.keras.layers.Conv2D(256,3,activation='relu',padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(num_classes,activation='softmax')
    ])
    return model

def CNN9(img_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=img_shape,padding='same'),
        tf.keras.layers.Conv2D(32,3,activation='relu',padding='same'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(64,3,activation='relu',padding='same'),
        tf.keras.layers.Conv2D(64,3,activation='relu',padding='same'),
        tf.keras.layers.Conv2D(64,3,activation='relu',padding='same'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(128,3,activation='relu',padding='same'),
        tf.keras.layers.Conv2D(128,2,activation='relu',padding='same'),
        tf.keras.layers.Conv2D(256,3,activation='relu',padding='same'),
        tf.keras.layers.Conv2D(256,3,activation='relu',padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(num_classes,activation='softmax')
    ])
    return model

def CNN10(img_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=img_shape,padding='same'),
        tf.keras.layers.Conv2D(128,3,activation='relu',padding='same'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(256,3,activation='relu',padding='same'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(num_classes,activation='softmax')
    ])
    return model

def CNN11(img_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=img_shape,padding='same'),
        tf.keras.layers.Conv2D(256,3,activation='relu',padding='same'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(256,3,activation='relu',padding='same'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(num_classes,activation='softmax')
    ])
    return model

def CNN11_NoPool(img_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=img_shape,padding='same'),
        tf.keras.layers.Conv2D(256,3,activation='relu',padding='same'),
        tf.keras.layers.Conv2D(256,3,activation='relu',padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(num_classes,activation='softmax')
    ])
    return model

def CNN12(img_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=img_shape,padding='same'),
        tf.keras.layers.Conv2D(512,3,activation='relu',padding='same'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(256,3,activation='relu',padding='same'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(num_classes,activation='softmax')
    ])
    return model

def test_CNN(img_shape, num_classes):
    #make a shallow CNN
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model

def Dense1(img_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=img_shape),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(num_classes,activation='softmax')
    ])
    return model

def Dense1_256(img_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=img_shape),
        tf.keras.layers.Dense(256,activation='relu'),
        tf.keras.layers.Dense(num_classes,activation='softmax')
    ])
    return model

def Dense1_512(img_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=img_shape),
        tf.keras.layers.Dense(512,activation='relu'),
        tf.keras.layers.Dense(num_classes,activation='softmax')
    ])
    return model

def Dense1_1024(img_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=img_shape),
        tf.keras.layers.Dense(1024,activation='relu'),
        tf.keras.layers.Dense(num_classes,activation='softmax')
    ])
    return model

def Dense1_2048(img_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=img_shape),
        tf.keras.layers.Dense(2048,activation='relu'),
        tf.keras.layers.Dense(num_classes,activation='softmax')
    ])
    return model

def Dense2(img_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=img_shape),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(num_classes,activation='softmax')
    ])
    return model

def Dense3(img_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=img_shape),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(num_classes,activation='softmax')
    ])
    return model

def Dense4(img_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=img_shape),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(num_classes,activation='softmax')
    ])
    return model


def get_model(model_name,img_shape, num_classes):
    if model_name == "CNN3":            return CNN3(img_shape, num_classes)
    elif model_name == "CNN4":          return CNN4(img_shape, num_classes)
    elif model_name == "CNN5":          return CNN5(img_shape, num_classes)
    elif model_name == "CNN5_NoPool":   return CNN5_NoPool(img_shape, num_classes)
    elif model_name == "CNN5_Dense1":   return CNN5_Dense1(img_shape, num_classes)
    elif model_name == "CNN5_Dense2":   return CNN5_Dense2(img_shape, num_classes)
    elif model_name == "CNN5_Dense3":   return CNN5_Dense3(img_shape, num_classes)
    elif model_name == "CNN5_Dense4":   return CNN5_Dense4(img_shape, num_classes)
    elif model_name == "CNN5_Dense5":   return CNN5_Dense5(img_shape, num_classes)
    elif model_name == "CNN5_DenseL":   return CNN5_DenseL(img_shape, num_classes)
    elif model_name == "CNN5_DenseXL":  return CNN5_DenseXL(img_shape, num_classes)
    elif model_name == "CNN5_DenseXXL": return CNN5_DenseXXL(img_shape, num_classes)
    elif model_name == "CNN6":          return CNN6(img_shape, num_classes)
    elif model_name == "CNN7":          return CNN7(img_shape, num_classes)
    elif model_name == "CNN8":          return CNN8(img_shape, num_classes)
    elif model_name == "CNN9":          return CNN9(img_shape, num_classes)
    elif model_name == "CNN10":         return CNN10(img_shape, num_classes)
    elif model_name == "CNN11":         return CNN11(img_shape, num_classes)
    elif model_name == "CNN11_NoPool":  return CNN11_NoPool(img_shape, num_classes)
    elif model_name == "CNN12":         return CNN12(img_shape, num_classes)
    elif model_name == "test_CNN":      return test_CNN()
    elif model_name == "Dense1":        return Dense1(img_shape, num_classes)
    elif model_name == "Dense1_256":    return Dense1_256(img_shape, num_classes)
    elif model_name == "Dense1_512":    return Dense1_512(img_shape, num_classes)
    elif model_name == "Dense1_1024":   return Dense1_1024(img_shape, num_classes)
    elif model_name == "Dense1_2048":   return Dense1_2048(img_shape, num_classes)
    elif model_name == "Dense2":        return Dense2(img_shape, num_classes)
    elif model_name == "Dense3":        return Dense3(img_shape, num_classes)
    elif model_name == "Dense4":        return Dense4(img_shape, num_classes)
    else: 
        print('Model not found')
        return None