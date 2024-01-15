
import numpy as np
import pandas as pd
import tensorflow as tf
import wandb
import os

os.environ['WANDB_API_KEY'] = 'fc2ea89618ca0e1b85a71faee35950a78dd59744'
wandb.login()
wandb.init(project="Tab_FIM")

#train a network on the titanic data to predict survival

#read the titanic data from file into a pandas dataframe
data_dir = 'datasets/TITANIC/data.csv'
titanic = pd.read_csv(data_dir)

#drop the columns that are not useful for prediction
titanic.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

#drop rows with missing values
titanic.dropna(inplace=True)

#scale the continuous features between -1 and 1
titanic['Age'] = (titanic['Age'] - titanic['Age'].mean()) / titanic['Age'].std()
titanic['Fare'] = (titanic['Fare'] - titanic['Fare'].mean()) / titanic['Fare'].std()

#convert categorical features to one-hot encoding
titanic = pd.get_dummies(titanic, columns=['Pclass', 'Sex', 'Embarked'])

#make the categorical columns 0 or 1
#print all headers
print(titanic.columns.values)
titanic['Pclass_1'] = titanic['Pclass_1'].astype('float32')
titanic['Pclass_2'] = titanic['Pclass_2'].astype('float32')
titanic['Pclass_3'] = titanic['Pclass_3'].astype('float32')
titanic['Sex_female'] = titanic['Sex_female'].astype('float32')
titanic['Sex_male'] = titanic['Sex_male'].astype('float32')
titanic['Embarked_C'] = titanic['Embarked_C'].astype('float32')
titanic['Embarked_Q'] = titanic['Embarked_Q'].astype('float32')
titanic['Embarked_S'] = titanic['Embarked_S'].astype('float32')
titanic['Survived'] = titanic['Survived'].astype('float32')
titanic['Parch'] = titanic['Parch'].astype('float32')
titanic['SibSp'] = titanic['SibSp'].astype('float32')
titanic['Age'] = titanic['Age'].astype('float32')
titanic['Fare'] = titanic['Fare'].astype('float32')



#slip into training 0.8 and test sets 0.2
titanic_train = titanic.iloc[:int(titanic.shape[0]*0.8)].copy()
titanic_test = titanic.iloc[int(titanic.shape[0]*0.8):].copy()

print(titanic_train.head())
print(titanic_test.head())

#print the dtype of each column
print(titanic_train.dtypes)

#convert the pandas dataframe to a tf dataset
titanic_train_x = titanic_train.iloc[:, 1:]
titanic_train_y = titanic_train.iloc[:, 0]
titanic_test_x = titanic_test.iloc[:, 1:]
titanic_test_y = titanic_test.iloc[:, 0]

#convert the pandas dataframe to a tf dataset
train_ds = tf.data.Dataset.from_tensor_slices((titanic_train_x, titanic_train_y))
test_ds = tf.data.Dataset.from_tensor_slices((titanic_test_x, titanic_test_y))

#build a model using the functional API
# Define the input layer
# inputs = tf.keras.layers.Input(shape=(12,))

# # Hidden layers
# x = tf.keras.layers.Dense(128, activation='relu')(inputs)
# x = tf.keras.layers.Dense(32, activation='relu')(x)

# # Output layer
# outputs = tf.keras.layers.Dense(1)(x)

# # Define the model
# model = tf.keras.Model(inputs=inputs, outputs=outputs)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(12,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.summary()

# Convert the tf.data datasets to batched datasets
batch_size = 8
titanic_train_ds = train_ds.shuffle(1000).batch(batch_size)
titanic_test_ds = test_ds.batch(batch_size)

#print the first batch of the training set
for x, y in train_ds.take(1):
    print(x.shape)
    print(y.shape)


# Compile the model
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.9)
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

# Train the model

for current_epoch in range(350):
    print('Epoch: ', current_epoch)
    history = model.fit(titanic_train_ds, validation_data=titanic_test_ds,epochs=1)

    wandb.log({"loss":history.history['loss'][-1],
        "val_loss":history.history['val_loss'][-1],
        "accuracy":history.history['binary_accuracy'][-1],
        "val_accuracy":history.history['val_binary_accuracy'][-1]}, step=current_epoch)

    def filter_dataset_by_loss(dataset, model, loss_function, threshold_percentage, less_than=True):
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
        
        print("c",c)

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
        c=0
        for _,_,_ in filtered_dataset:
            c+=1
        print("fc",c)

        # If you want to remove the loss from the dataset after filtering
        filtered_dataset = filtered_dataset.map(lambda loss, inputs, targets: (inputs, targets))
        filtered_dataset = filtered_dataset.unbatch()
        filtered_dataset = filtered_dataset.batch(batch_size)

        return filtered_dataset, c

    filtered_dataset, new_d_count = filter_dataset_by_loss(titanic_train_ds,
        model,
        tf.keras.losses.BinaryCrossentropy(from_logits=True),
        90,
        less_than=False)

    iter_ds = iter(filtered_dataset)

    max_fim_batches = 100
    if new_d_count//batch_size > max_fim_batches: new_d_count = max_fim_batches
    FIM_batches = new_d_count//batch_size
    x_mean = 0
    c = 0

    

    for _ in range(FIM_batches):
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
    wandb.log({"FIM_high": x_mean}, step=current_epoch)  

    filtered_dataset,new_d_count = filter_dataset_by_loss(titanic_train_ds,
        model,
        tf.keras.losses.BinaryCrossentropy(from_logits=True),
        10,
        less_than=True)

    iter_ds = iter(filtered_dataset)

    max_fim_batches = 100
    if new_d_count//batch_size > max_fim_batches: new_d_count = max_fim_batches
    FIM_batches = new_d_count//batch_size
    #loop batches
    x_mean = 0
    c = 0
    for _ in range(FIM_batches):
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
    wandb.log({"FIM_low": x_mean}, step=current_epoch) 








