import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_datasets as tfds
import math
import os
import wandb

#wandb init
os.environ['WANDB_API_KEY'] = 'fc2ea89618ca0e1b85a71faee35950a78dd59744'
wandb.login()
wandb.init(project="Pre_Training_GFIM")

#load the mnist dataset
train_ds = tfds.load('mnist', split='train', shuffle_files=True)
test_ds = tfds.load('mnist', split='test', shuffle_files=True)

batch_size = 32
seed = 42

#preprocess the data
def preprocess(features):
    image = tf.cast(features['image'], tf.float32) / 255.
    label = tf.cast(features['label'], tf.int32)
    return image, label

def random_labels(image, label):
    label = tf.random.uniform(shape=[], minval=0, maxval=9, dtype=tf.int32)
    return image, label

def scale(image, label):
    image = tf.image.resize(image, (299, 299))
    #convert the image to have 3 channels
    image = tf.repeat(image, 3, -1)
    return image, label

#calculate the memory usage of the dataset
def get_dataset_memory_usage_in_bytes(dataset):
    # Iterate over the dataset and calculate its memory usage in bytes
    total_memory = 0
    for element in dataset:
        _, labels = element
        total_memory += labels.numpy().nbytes
    return total_memory


train_ds = train_ds.map(preprocess).map(scale).map(random_labels)
print("Memory usage in bytes: ", get_dataset_memory_usage_in_bytes(train_ds))

train_ds = train_ds.shuffle(1024).batch(batch_size)
train_ds = train_ds.cache().prefetch(tf.data.experimental.AUTOTUNE)

test_ds = test_ds.map(preprocess).map(scale)

train_ds = train_ds.cache().prefetch(tf.data.experimental.AUTOTUNE)

# build a IrV2 model without the last 28 layers for mnist dataset
irv2 = tf.keras.applications.InceptionResNetV2(
                include_top=True,
                weights=None,
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classifier_activation="softmax",
            )
irv2_out = irv2.layers[-28].output
flatten = tf.keras.layers.Flatten()(irv2_out)
output = tf.keras.layers.Dense(10, activation='softmax')(flatten)
model = tf.keras.models.Model(inputs=irv2.input, outputs=output)


#compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

#model.summary()
#count number of layers in model
print("Number of Layers:",len(model.layers))

#print model summary of first layer
print("Inp Shape:",model.layers[0].input.shape)

#train the model
def filter_dataset_by_loss(dataset, model, loss_function, threshold_percentage_low,threshold_percentage_high,new_bs,max_data=5000):
    # Compute the losses for all examples and store them in a new dataset
    # de batch the dataset
    dataset = dataset.unbatch()
    dataset = dataset.batch(1)
    losses_and_data = []
    c=0
    for inputs, targets in dataset:
        predictions = model(inputs,training=False)
        loss = loss_function(targets, predictions)
        c+=1
        losses_and_data.append((loss.numpy(), inputs, targets))
    
    # Convert to a tf.data.Dataset
    losses_and_data_dataset = tf.data.Dataset.from_generator(
        lambda: losses_and_data, output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.float32),
            tf.TensorSpec(shape=inputs.shape, dtype=tf.float32),
            tf.TensorSpec(shape=targets.shape, dtype=tf.float32),
        )
    )

    # Calculate the threshold for the lowest threshold_percentage of the loss
    losses = [item[0] for item in losses_and_data]
    threshold_low = np.percentile(losses, threshold_percentage_low)
    threshold_high = np.percentile(losses, threshold_percentage_high)

    # Function to filter the dataset
    def filter_func(loss, inputs, targets):
        return loss <= threshold_high and loss >= threshold_low

    # Filter the dataset
    filtered_dataset = losses_and_data_dataset.filter(filter_func)
    c=0
    for _,_,_ in filtered_dataset:
        c+=1
    if c > max_data:
        filtered_dataset = filtered_dataset.take(max_data)
        c = max_data
    new_batch_count = math.ceil(c/new_bs)

    # If you want to remove the loss from the dataset after filtering
    filtered_dataset = filtered_dataset.map(lambda loss, inputs, targets: (inputs, targets))
    filtered_dataset = filtered_dataset.unbatch()
    filtered_dataset = filtered_dataset.batch(new_bs)
    return filtered_dataset, c, new_batch_count

@tf.function
def Get_Z(items):
    imgs,labels = items
    with tf.GradientTape() as tape:
        y_hat = self.model(imgs,training=False)
        selected = tf.squeeze(tf.random.categorical(tf.math.log(y_hat), 1))
        output = tf.gather(y_hat,selected,axis=1,batch_dims=1)
        output = tf.math.log(output)
    g = tape.jacobian(output,self.model.trainable_variables)
    layer_sizes = [tf.reduce_sum(tf.size(v)) for v in self.model.trainable_variables]
    g = [tf.reshape(g[i],(self.FIM_BS,layer_sizes[i])) for i in range(len(g))]
    g = tf.concat(g,axis=1)
    g = tf.square(g)
    g = tf.reduce_sum(g,axis=1)
    return g #tensor of shape (bs)

current_epoch = 1
epochs = 2
FIM_BS = 2

while current_epoch <= epochs:
    print("Epoch", current_epoch)
    bc = 0
    history = model.fit(train_ds,test_ds, epochs=1, verbose=2)

    wandb.log({"loss":history.history['loss'][-1],
        "val_loss":history.history['val_loss'][-1],
        "accuracy":history.history['sparse_categorical_accuracy'][-1],
        "val_accuracy":history.history['val_sparse_categorical_accuracy'][-1]}, step=current_epoch)
    
    #Filter dataset by loss
    groups = 8  #number of groups to split the dataset
    for n in range(groups): 
        low_perc = int(100/groups)*n
        high_perc = int(100/groups)*(n+1)

        filtered_dataset,c new_batch_count = filter_dataset_by_loss(train_ds,
                                    model,
                                    tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                                    low_perc,
                                    high_perc,
                                    FIM_BS,
                                    max_data=10) #new batch size for FIM
        print("Group:",n,"Batches:",new_batch_count,"Data Points:",c)

        x_mean = 0
        c = 0
        iter_ds = iter(filtered_dataset)
        for _ in range(new_batch_count):
            if data_count/new_batch_count % 500 == 0:
                print(c)
            FIM_estimates = Get_Z(next(iter_ds))#returns a batch of FIM estimates
            c += FIM_BS
            deltas = FIM_estimates - x_mean * FIM_BS
            x_mean += tf.reduce_sum(deltas, axis=0) / c

        print("FIM", x_mean)
        wandb.log({"FIM_"+str(n): x_mean}, step=current_epoch)  
    
    current_epoch += 1

model.save("model.h5")

