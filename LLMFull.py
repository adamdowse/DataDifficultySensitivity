#all code in one file to be slip out later

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers,losses
import os
import string
import re
import shutil

import wandb

#wandb init
os.environ['WANDB_API_KEY'] = 'fc2ea89618ca0e1b85a71faee35950a78dd59744'
wandb.login()
wandb.init(project="LLM_FIM")



batch_size = 32
seed = 42
max_features = 10000
sequence_length = 250

#download the data
if False:
    url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    dataset = tf.keras.utils.get_file('aclImdb_v1.tar.gz', url, untar=True, cache_dir='datasets/IMDB', cache_subdir='')
    dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
    train_dir = os.path.join(dataset_dir, 'train')
    remove_dir = os.path.join(train_dir, 'unsup')
    shutil.rmtree(remove_dir)
else:
    dataset_dir = 'datasets/IMDB/aclImdb'
    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')
    print("Dataset dir", dataset_dir)


raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)

raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    train_dir, 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='validation', 
    seed=seed)

raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    test_dir, 
    batch_size=batch_size)

print("Label 0 corresponds to", raw_train_ds.class_names[0])
print("Label 1 corresponds to", raw_train_ds.class_names[1])

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')

max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

# retrieve a batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print("Review", first_review)
print("Label", raw_train_ds.class_names[first_label])
print("Vectorized review", vectorize_text(first_review, first_label))

print("1287 ---> ",vectorize_layer.get_vocabulary()[1287])
print(" 313 ---> ",vectorize_layer.get_vocabulary()[313])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

embedding_dim = 16

# model = tf.keras.Sequential([
#     layers.Embedding(max_features, embedding_dim),
#     layers.Dropout(0.2),
#     layers.GlobalAveragePooling1D(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(32, activation='relu'),
#     layers.Dense(16, activation='relu'),
#     layers.Dense(8, activation='relu'),
#     layers.Dropout(0.2),
#     layers.Dense(1)])

model = tf.keras.Sequential([
    layers.Embedding(max_features + 1, embedding_dim, input_length=sequence_length),
    layers.Conv1D(128, 5, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 5, activation='relu'),
    layers.Dropout(0.2),
    layers.MaxPooling1D(2),
    layers.Conv1D(32, 5, activation='relu'),
    layers.Dropout(0.2),
    layers.GlobalMaxPooling1D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

model.summary()

#optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.5)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
                optimizer=optimizer,
                metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

epochs = 100
current_epoch = 1

while current_epoch <= epochs:
    print("Epoch", current_epoch)
    bc = 0
    tloss = 0
    taccuracy = 0
    
    # for text_batch, label_batch in train_ds:
    #     bc += 1
    #     loss, accuracy = model.train_on_batch(text_batch, label_batch)
    #     tloss += loss
    #     taccuracy += accuracy
    # print("Epoch", current_epoch, "Loss", tloss/bc, "Accuracy", taccuracy/bc)
    #wandb.log({"loss": tloss/bc, "accuracy": taccuracy/bc}, step=current_epoch)

    # bc = 0
    # tval_loss = 0
    # tval_accuracy = 0
    # for text_batch, label_batch in val_ds:
    #     bc += 1
    #     val_loss, val_accuracy = model.evaluate(text_batch, label_batch)
    #     tval_loss += val_loss
    #     tval_accuracy += val_accuracy
    # print("Epoch", current_epoch, "Val Loss", tval_loss/bc, "Val Accuracy", tval_accuracy/bc)
    # wandb.log({"val_loss": tval_loss/bc, "val_accuracy": tval_accuracy/bc}, step=current_epoch)
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=1)
    wandb.log({"loss":history.history['loss'][-1],
        "val_loss":history.history['val_loss'][-1],
        "accuracy":history.history['binary_accuracy'][-1],
        "val_accuracy":history.history['val_binary_accuracy'][-1]}, step=current_epoch)

    #perfrom the FIM analysis

    #create a new dataset that contains only the lowest 5% of the loss
    def filter_dataset_by_loss(dataset, model, loss_function, threshold_percentage, less_than=True):
        # Compute the losses for all examples and store them in a new dataset
        losses_and_data = []
        for inputs, targets in dataset:
            predictions = model(inputs,training=False)
            loss = loss_function(targets, predictions)
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
        threshold = np.percentile(losses, threshold_percentage)

        # Function to filter the dataset
        def filter_func(loss, inputs, targets):
            if less_than:
                return loss <= threshold
            else:
                return loss >= threshold

        # Filter the dataset
        filtered_dataset = losses_and_data_dataset.filter(filter_func)

        # If you want to remove the loss from the dataset after filtering
        filtered_dataset = filtered_dataset.map(lambda loss, inputs, targets: (inputs, targets))

        return filtered_dataset

    filtered_dataset = filter_dataset_by_loss(train_ds, model, losses.BinaryCrossentropy(from_logits=True), 95, less_than=False)

    iter_ds = iter(filtered_dataset)
    #loop batches
    x_mean = 0
    c = 0
    for _ in range(10):
        text_batch, label_batch = next(iter_ds)
        for i in range(len(text_batch)):
            if c % 100 == 0:
                print(c)
            with tf.GradientTape() as tape:
                #print(text_batch[i])
                item = tf.expand_dims(text_batch[i],0)
                y_hat = model(item,training=False) #[0.2]
                y_hat = tf.nn.sigmoid(y_hat)                #convert to probabilities [0.3]
                y_hat = tf.concat([1-y_hat,y_hat],axis=1) #[0.3,0.7]  #convert to categorical
                selected = tf.squeeze(tf.random.categorical(tf.math.log(y_hat),1),axis=0) #DO I NEED TO LOG THIS Y_HAT?
                output = tf.gather(y_hat,selected,axis=1) #Check dimentions
                output = tf.math.log(output)

            g = tape.gradient(output,model.trainable_variables)
            g = [tf.square(i) for i in g]
            g = [tf.reduce_sum(i) for i in g]
            g = tf.reduce_sum(g)
            
            c += 1
            x_mean = x_mean + (g - x_mean)/c
            if g == 0:
                print("ZERO grad found", x_mean)

    print("FIM", x_mean)
    wandb.log({"FIM": x_mean}, step=current_epoch)  

    current_epoch += 1

#history = model.fit(
#    train_ds,
#    validation_data=val_ds,
#    epochs=epochs)

loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)