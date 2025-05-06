import tensorflow as tf
import numpy as np
import wandb
import os






def main():
    # Load the data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    

    x_train = x_train/255
    x_test = x_test/255

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)) 
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    def map_fn(image, label):
        #image = tf.cast(image, tf.float32)
        #image = tf.expand_dims(image, -1)
        return image, tf.squeeze(tf.one_hot(tf.cast(label,tf.int32), 10,on_value=1.0))


    train_dataset = train_dataset.map(map_fn).shuffle(10000).batch(32)
    test_dataset = test_dataset.map(map_fn).batch(32)

    train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Load the model
    inp = tf.keras.layers.Input(shape=(32,32,3))
    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(inp)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(inputs=inp, outputs=x)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train)
            loss_values = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_batch_train)
        #tf.print("loss_values")
        #tf.print(loss_values)
        
        jacobian = tape.jacobian(loss_values, model.trainable_weights)

        #compute the variance over the batch
        variance = []
        grad = []
        for layer in jacobian:
            v = tf.math.reduce_variance(layer,axis=0)+1e-8
            v = v/tf.reduce_max(v)
            variance.append(v)
            grad.append(tf.reduce_mean(layer, axis=0)*v)

        loss_value = tf.reduce_mean(loss_values)

        model.optimizer.apply_gradients(zip(grad, model.trainable_weights))
        return loss_value

    @tf.function
    def alpha_step(x,y,fisher_buffer):
        with tf.GradientTape() as tape:
            logits = model(x)
            loss_value = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
        gradient = tape.gradient(loss_value, model.trainable_weights)

        #fisher for the batch
        F = [g**2 for g in gradient]
        
        #if fisher buffer is not full
        buffer_size = 10
        if len(fisher_buffer) < buffer_size:
            fisher_buffer.append(F)
        else:
            fisher_buffer.pop(0)
            fisher_buffer.append(F)




    epochs = 10



    for epoch in range(epochs):
        loss_value = 0
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            step_loss = train_step(x_batch_train, y_batch_train)
            #print(step_loss)
            loss_value += step_loss
            if step % 100 == 0:
                print("Training loss (for one batch) at step %d: %.4f" % (step, float(loss_value/(step+1))))
                print("Seen so far: %s samples" % ((step + 1) * 32))
        
        training_loss = loss_value / step
        print("Training loss: %.4f" % (float(training_loss)))
        wandb.log({"Training Loss": training_loss})

        # Run a validation loop at the end of each epoch.
        step_loss = 0
        for step,(x_batch_val, y_batch_val) in enumerate(test_dataset):
            val_logits = model(x_batch_val)
            step_loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=val_logits, labels=y_batch_val))
        print("Validation loss: %.4f" % (float(step_loss/(step+1))))
        wandb.log({"Validation Loss": step_loss/(step+1)})



if __name__ == "__main__":
    os.environ['WANDB_API_KEY'] = 'fc2ea89618ca0e1b85a71faee35950a78dd59744'
    wandb.login()
    wandb.init(project="RMSBatch",name="a)SGD")
    main()