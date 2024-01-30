import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_datasets as tfds
import math



#load the mnist dataset
train_ds = tfds.load('mnist', split='train', shuffle_files=True)
test_ds = tfds.load('mnist', split='test', shuffle_files=True)

batch_size = 32
seed = 42

#preprocess the data
def preprocess(features):
    image = tf.cast(features['image'], tf.float32) / 255.
    label = tf.random.uniform(shape=[], minval=0, maxval=9, dtype=tf.int64)
    return image, label

train_ds = train_ds.map(preprocess).shuffle(1024).batch(batch_size)

train_ds = train_ds.cache().prefetch(tf.data.experimental.AUTOTUNE)

#build a vision transformer model
def build_VIT(new_img_size=72,patch_size=6,projection_dim=64,num_heads=4,transformer_layers=8,mlp_head_units=[2048,1024]):

    num_patches = (new_img_size // patch_size) ** 2
    transformer_units = [projection_dim * 2,projection_dim]

    def mlp(x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = tf.keras.layers.Dense(units, activation=tf.keras.activations.gelu)(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)
        return x

    class Patches(keras.layers.Layer):
        def __init__(self, patch_size):
            super().__init__()
            self.patch_size = patch_size
        def call(self, images):
            input_shape = tf.shape(images)
            batch_size = input_shape[0]
            height = input_shape[1]
            width = input_shape[2]
            channels = input_shape[3]
            num_patches_h = height // self.patch_size
            num_patches_w = width // self.patch_size
            patches = tf.image.extract_patches(images=images,sizes=[1,self.patch_size,self.patch_size,1],strides=[1,self.patch_size,self.patch_size,1],rates=[1,1,1,1],padding="VALID")
            #patches = keras.ops.image.extract_patches(images, size=self.patch_size)
            patches = tf.reshape(patches,(batch_size,num_patches_h * num_patches_w, self.patch_size * self.patch_size * channels))
            #patches = keras.ops.reshape(patches,(
            #    batch_size,
            #    num_patches_h * num_patches_w,
            #    self.patch_size * self.patch_size * channels,
            #))
            return patches
        def get_config(self):
            config = super().get_config()
            config.update({"patch_size": self.patch_size})
            return config
    class PatchEncoder(keras.layers.Layer):
        def __init__(self, num_patches, projection_dim):
            super().__init__()
            self.num_patches = num_patches
            self.projection = tf.keras.layers.Dense(units=projection_dim)
            self.position_embedding = tf.keras.layers.Embedding(input_dim=num_patches,output_dim=projection_dim)
        def call(self, patch):
            positions = tf.expand_dims(tf.range(0,self.num_patches),axis=0)
            #positions = keras.ops.expand_dims(keras.ops.arange(start=0,stop=self.num_patches,step=1),axis=0)
            projected_patches = self.projection(patch)
            encoded = projected_patches + self.position_embedding(positions)
            return encoded
        def get_config(self):
            config = super().get_config()
            config.update({"num_patches": self.num_patches})
            return config

    data_aug = tf.keras.Sequential(
        [
            #tf.keras.layers.Normalization(),
            tf.keras.layers.Resizing(new_img_size,new_img_size),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(factor=0.02),
            tf.keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2),
        ], name="data_augmentation")

    inputs = tf.keras.Input(shape=self.img_shape)
    augmented = data_aug(inputs)
    patches = Patches(patch_size)(augmented)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    for _ in range(transformer_layers):
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads,key_dim=projection_dim, dropout=0.1)(x1,x1)
        x2 = tf.keras.layers.Add()([attention_output,encoded_patches])
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        encoded_patches = tf.keras.layers.Add()([x3,x2])
    representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = tf.keras.layers.Flatten()(representation)
    representation = tf.keras.layers.Dropout(0.5)(representation)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    logits = tf.keras.layers.Dense(self.num_classes,activation="softmax")(features)
    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model

model = build_VIT()

#compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

#train the model
model.fit(train_ds, epochs=50, verbose=2)


