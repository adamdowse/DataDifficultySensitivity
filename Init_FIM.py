#Record the initial FIM and the final test accuracy of the model.

import wandb 
import os



def main():
    api = wandb.Api()
    sweep = api.sweep(f"adamdowse/Model_Initialisation/CNN0") #This is the sweep id from the sweep config file

    # Get best run parameters
    best_run = sweep.best_run()
    best_parameters = best_run.config
    lr = best_parameters['lr']
    bs = best_parameters['bs']

    max_epochs = 30

    #setup data 
    #get mnist data from tfds
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    #convert to tfds
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.int64)
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.int64)

    #noramlise and build the datasets
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(lambda x, y: (x / 255, y))
    train_ds = train_ds.shuffle(60000).batch(bs)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(lambda x, y: (x / 255, y))
    test_ds = test_ds.batch(bs)

    #pull model from seperate file
    model = im.get_model("CNN0")

    #compile the model
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    #callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    def Get_Z_single(item):
        img,label,loss = item
        with tf.GradientTape() as tape:
            y_hat = model(img,training=False) #[0.1,0.8,0.1,ect] this is output of softmax
            output = tf.squeeze(tf.random.categorical(tf.math.log(y_hat), 1)) #[2]
            #output = tf.gather(y_hat,selected,axis=1,batch_dims=1)
            output = tf.gather(y_hat,output,axis=1) #[0.3]
            output = tf.squeeze(output)
            output = tf.math.log(output)
        g = tape.gradient(output,model.trainable_variables)#This or Jacobian?
        
        layer_sizes = [tf.reduce_sum(tf.size(v)) for v in model.trainable_variables]
        g = [tf.reshape(g[i],(layer_sizes[i])) for i in range(len(g))] #TODO check that this dosent need to deal with batches
        g = tf.concat(g,axis=0)
        g = tf.square(g)
        g = tf.reduce_sum(g)
        return g

    def record_GFIM(ds,model):
        print('recording GFIM')
        ds = ds.unbatch()
        ds = ds.batch(1)
        data_count = 0
        mean = 0
        iter_ds = iter(ds)
        low_lim = 2000
        for _ in range(low_lim):
            data_count += 1
            if data_count % 500 == 0:
                print(data_count)
            x = Get_Z_single(next(iter_ds)) #just one replica can be used here
            delta = x - mean 
            mean += delta/(data_count)
        wandb.log({'GFIM':mean},step=0)
        return

    #record the initial FIM
    record_GFIM(train_ds,model)

    hist = model.fit(train_ds,
        epochs=max_epochs,
        validation_data=test_ds,
        callbacks=[early_stopping])

    #log max val accuracy
    wandb.log({'Max_Val_Acc':max(hist.history['val_accuracy'])},step=0)
    return 

if __name__ == "__main__":
    os.environ['WANDB_API_KEY'] = 'fc2ea89618ca0e1b85a71faee35950a78dd59744'
    wandb.login()
    main()



