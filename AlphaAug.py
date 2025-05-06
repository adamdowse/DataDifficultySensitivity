


#This will setup the automatic augmentaion of the data the data based on the maximum allpha value produced for a the aug subset.



import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
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
    train_dataset = train_dataset.map(map_fn)
    test_dataset = test_dataset.map(map_fn)
    train_dataset = train_dataset.shuffle(50000).batch(32)
    test_dataset = test_dataset.batch(32)
    train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Load the model
    # define noise variable to it can be changed during training
    noise_val =  tf.keras.backend.variable(np.array(0.0, dtype=np.float32), dtype=tf.float32, name='noise')
    
    class AugLayer(tf.keras.layers.Layer):
        def __init__(self, initial_noise):
            super(AugLayer, self).__init__()
            self.inital_noise = initial_noise
            self.noise = tf.Variable(initial_value=self.inital_noise, trainable=False,name='noise')

        # def build(self, input_shape):
        #     self.kernel = self.add_weight("kernel",
        #                                 shape=[int(input_shape[-1]),
        #                                         self.num_outputs])

        def call(self, inputs):
            return tf.keras.layers.GaussianNoise(self.noise)(inputs)
    
    auglayer = AugLayer(0.0)

    inp = tf.keras.layers.Input(shape=(32,32,3))
    x = auglayer(inp)
    x = tf.clip_by_value(x, 0, 1)
    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(inputs=inp, outputs=x)

    #print the variables in the auglayer
    print(model.variables[0])


    tf.keras.backend.set_value(model.variables[0], 0.0)

    # Compile the model
    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

    #callback to update noise variable
    class UpdateNoise(tf.keras.callbacks.Callback):
        def __init__(self,dataset,max_data):
            self.dataset = dataset
            self.max_data = max_data
            self.noise_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            self.loss_function = tf.keras.losses.CategoricalCrossentropy()
        def on_epoch_begin(self, epoch, logs=None):
            def func(x,a):
                return a*np.log(1+x)**2
            # calculate noise based on alpha value
            data = [] #[(noise_level, alpha)]
            for noise_level in self.noise_levels:
                #set the noise variable in the model
                
                self.model.variables[0].assign(noise_level)
                tf.print(self.model.variables[0])
            
                c = 0 
                for items in self.dataset.take(self.max_data):
                    if c > self.max_data:
                        break
                    j, y_hat = self.Get_Z_sm_uniform(items)
                    if c == 0:
                        FIMs = j
                        Y_hats = y_hat
                    else:
                        FIMs = tf.concat([FIMs,j],axis=0)
                        Y_hats = tf.concat([Y_hats,y_hat],axis=0)
                    c += 1
            
                FIMs = tf.squeeze(FIMs)
                Y_hats = tf.squeeze(Y_hats)

                popt, pcov = curve_fit(func, Y_hats.numpy(),FIMs.numpy())
                data.append((noise_level,popt[0]))
                tf.print(noise_level,popt[0])
            #find the max alpha value
            max_alpha = max(data,key=lambda x: x[1])
            self.model.variables[0].assign(max_alpha[0])
            tf.print("Max Alpha: ",max_alpha)
            wandb.log({'noise':max_alpha[0],
                        'alpha_selected':max_alpha[1],
                        "alpha0":data[0][1],
                        "alpha10":data[1][1],
                        "alpha20":data[2][1],
                        "alpha30":data[3][1],
                        "alpha40":data[4][1],
                        "alpha50":data[5][1],
                        "alpha60":data[6][1],
                        "alpha70":data[7][1],
                        "alpha80":data[8][1],
                        "alpha90":data[9][1],
                        "alpha100":data[10][1]})


            
        @tf.function
        def Get_Z_sm_uniform(self,items):
            x,y = items
            bs = tf.shape(x)[0]
            
            with tf.GradientTape() as tape:
                #print(text_batch[i])
                y_hat = self.model(x,training=False) #get model output (softmax) [BS x num_classes]
                loss = self.loss_function(y,y_hat)
                selected = tf.squeeze(tf.random.uniform((bs,),0,10,dtype=tf.int64)) #sample from the output [BS x 1]
                y_hat = tf.gather(y_hat,selected,axis=1,batch_dims=1) #get the output for the selected class [BS x 1]
                output = tf.math.log(y_hat)#tf.math.log(output) #log the output [BS x 1]

            j = tape.jacobian(output,self.model.trainable_variables)
            layer_sizes = [tf.reduce_sum(tf.size(v)) for v in self.model.trainable_variables] #get the size of each layer
            j = [tf.reshape(j[i],(bs,layer_sizes[i])) for i in range(len(j))] #reshape the gradient to [BS x num_layer_params x layers]
            j = tf.concat(j,axis=1) #concat the gradient over the layers [BS x num_params]
            j = tf.square(j) #square the gradient [BS x num_params]
            j = tf.reduce_sum(j,axis=1) #sum the gradient [ 1]
            return j, y_hat
            
    
    alphaNoise = UpdateNoise(train_dataset,100)
    WandbCallback = wandb.keras.WandbCallback()

    # Train the model
    model.fit(train_dataset, epochs=10, callbacks=[alphaNoise,WandbCallback])





if __name__ == "__main__":
    os.environ['WANDB_API_KEY'] = 'fc2ea89618ca0e1b85a71faee35950a78dd59744'
    wandb.login()
    wandb.init(project="AlphaAug",name="Test")

    main()
    wandb.finish()