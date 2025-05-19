


#This will setup the automatic detection of outliers.



import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import wandb
import os



def main():
    # Load the data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    

    # x_train = x_train/255
    x_test = x_test/255

    # train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)) 
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    def map_fn(image, label):
        #image = tf.cast(image, tf.float32)
        #image = tf.expand_dims(image, -1)
        return image, tf.squeeze(tf.one_hot(tf.cast(label,tf.int32), 10,on_value=1.0))


    # train_dataset = train_dataset.map(map_fn)
    test_dataset = test_dataset.map(map_fn)
    test_dataset = test_dataset.batch(32)

    # train_dataset = train_dataset.shuffle(50000).batch(32)
    # test_dataset = test_dataset.batch(32)
    # train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
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


    # Compile the model
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    WandbCallback = wandb.keras.WandbCallback(save_model=False)

    class DataGen():
        def __init__(self,X,Y):
            self.X = X
            self.Y = Y
            self.X = self.X/255
            self.Y = tf.squeeze(tf.one_hot(tf.cast(self.Y,tf.int32), 10,on_value=1.0))
            print(self.Y)
            self.mask = np.ones(len(X),dtype=bool)

        def scale(self,x):
            return x/255
        
        def one_hot(self,y):
            return tf.squeeze(tf.one_hot(tf.cast(y,tf.int32), 10,on_value=1.0))

        def __call__(self):
            print(np.sum(self.mask))
            if np.sum(self.mask) >= 32:
                #get batch indexes
                idxs = np.random.choice(np.where(self.mask)[0],32)
                x = self.X[idxs]
                y = self.Y[idxs]
                self.mask[idxs] = False
                print(x.shape)
                return x,y
            else:
                return None,None
            
            
        
        @tf.function
        def Get_Z_sm_uniform_single(self,item,model):
            x,y = item
            x = tf.expand_dims(x,0)
            y = tf.expand_dims(y,0)
            bs = tf.shape(x)[0]
            loss_function = tf.keras.losses.CategoricalCrossentropy()
            
            with tf.GradientTape() as tape:
                #print(text_batch[i])
                y_hat = model(x,training=False) #get model output (softmax) [1 x num_classes]
                loss = loss_function(y,y_hat)
                selected = tf.random.uniform((bs,),0,10,dtype=tf.int64) #sample from the output [1 x 1]
                y_hat = tf.gather(y_hat,selected,axis=1,batch_dims=1) #get the output for the selected class [BS x 1]
                output = tf.math.log(y_hat)#tf.math.log(output) #log the output [BS x 1]

            j = tape.jacobian(output,model.trainable_variables)
            layer_sizes = [tf.reduce_sum(tf.size(v)) for v in model.trainable_variables] #get the size of each layer
            j = [tf.reshape(j[i],(bs,layer_sizes[i])) for i in range(len(j))] #reshape the gradient to [BS x num_layer_params x layers]
            j = tf.concat(j,axis=1) #concat the gradient over the layers [BS x num_params]
            j = tf.square(j) #square the gradient [BS x num_params]
            j = tf.reduce_sum(j,axis=1) #sum the gradient [ 1]
            return j, y_hat,output

        @tf.function
        def Get_Z_sm_sample_single_emp(self,item,model):
            x,y = item
            x = tf.expand_dims(x,0)
            y = tf.expand_dims(y,0)
            bs = tf.shape(x)[0]
            loss_function = tf.keras.losses.CategoricalCrossentropy()
            
            with tf.GradientTape() as tape:
                #print(text_batch[i])
                y_hat = model(x,training=False) #get model output (softmax) [1 x num_classes]
                loss = loss_function(y,y_hat)
                selected = tf.random.uniform((bs,),0,10,dtype=tf.int64) #sample from the output [1 x 1]
                #y_hat = tf.gather(y_hat,selected,axis=1,batch_dims=1) #get the output for the selected class [BS x 1]
                y_hat = tf.gather(y_hat,tf.argmax(y,axis=1),axis=1,batch_dims=1)
                output = tf.math.log(y_hat)#tf.math.log(output) #log the output [BS x 1]

            j = tape.jacobian(output,model.trainable_variables)
            layer_sizes = [tf.reduce_sum(tf.size(v)) for v in model.trainable_variables] #get the size of each layer
            j = [tf.reshape(j[i],(bs,layer_sizes[i])) for i in range(len(j))] #reshape the gradient to [BS x num_layer_params x layers]
            j = tf.concat(j,axis=1) #concat the gradient over the layers [BS x num_params]
            j = tf.square(j) #square the gradient [BS x num_params]
            j = tf.reduce_sum(j,axis=1) #sum the gradient [ 1]
            return j, y_hat,output
        
        @tf.function
        def Get_Z_sm_emp(self,item,model):
            x,y = item
            bs = tf.shape(x)[0]
            loss_function = tf.keras.losses.CategoricalCrossentropy()
            
            with tf.GradientTape() as tape:
                #print(text_batch[i])
                y_hat = model(x,training=False) #get model output (softmax) [1 x num_classes]
                loss = loss_function(y,y_hat)
                #selected = tf.random.uniform((bs,),0,10,dtype=tf.int64) #sample from the output [1 x 1]
                #y_hat = tf.gather(y_hat,selected,axis=1,batch_dims=1) #get the output for the selected class [BS x 1]
                y_hat = tf.gather(y_hat,tf.argmax(y,axis=1),axis=1,batch_dims=1)
                output = tf.math.log(y_hat)#tf.math.log(output) #log the output [BS x 1]

            j = tape.jacobian(output,model.trainable_variables)
            layer_sizes = [tf.reduce_sum(tf.size(v)) for v in model.trainable_variables] #get the size of each layer
            j = [tf.reshape(j[i],(bs,layer_sizes[i])) for i in range(len(j))] #reshape the gradient to [BS x num_layer_params x layers]
            j = tf.concat(j,axis=1) #concat the gradient over the layers [BS x num_params]
            j = tf.square(j) #square the gradient [BS x num_params]
            j = tf.reduce_sum(j,axis=1) #sum the gradient [ 1]
            return j, y_hat,output
        
        
        @tf.function
        def filter_func(self,x,y,alpha):
            def tf_func(x,a):
                return a*tf.math.log(1+x)**2
            j, y_hat,output = self.Get_Z_sm_sample_single_emp((x,y),model)

            return tf.squeeze(j - tf_func(-output,alpha) > 0)
        
        @tf.function
        def filter_func_batch(self,x,y,alpha):
            def tf_func(x,a):
                return a*tf.math.log(1+x)**2
            j, y_hat,output = self.Get_Z_sm_emp((x,y),model)
            return j - tf_func(-output,alpha) > 0
        
        def on_epoch_begin(self):
            #calc alpha 
            def func(x,a):
                return a*np.log(1+x)**2

            sampleFIMs = []
            yHats = []
            for i in range(500):
                x,y = self.X[i],self.Y[i]
                j, y_hat,output = self.Get_Z_sm_uniform_single((x,y),model)
                sampleFIMs.append(j)
                yHats.append(-output)
            sampleFIMs = tf.concat(sampleFIMs,axis=0)
            yHats = tf.concat(yHats,axis=0)
            popt, pcov = curve_fit(func, yHats.numpy(),sampleFIMs.numpy())
            self.alpha = tf.cast(popt[0],tf.float32)
            print("Alpha: ",self.alpha)
            wandb.log({"Alpha":self.alpha,"MeanSampleFIM":tf.reduce_mean(sampleFIMs)},commit=False)

            #filter the data
            temp_ds = tf.data.Dataset.from_tensor_slices((self.X,self.Y)).batch(32)
            bools = []
            c=0
            for items in temp_ds:
                c+=1
                print(c)
                x,y = items
                batch_bools = self.filter_func_batch(x,y,self.alpha).numpy()
                bools.append(batch_bools)
            bools = np.concatenate(bools,axis=0)
            self.mask = bools
            del temp_ds



            print("Filtered Dataset Length: ",np.sum(self.mask))
            wandb.log({"Filtered Dataset Length":np.sum(self.mask)},commit=False)
            


    @tf.function
    def Get_Z_sm_uniform(items,model):
        x,y = items
        bs = tf.shape(x)[0]
        loss_function = tf.keras.losses.CategoricalCrossentropy()
        
        with tf.GradientTape() as tape:
            #print(text_batch[i])
            y_hat = model(x,training=False) #get model output (softmax) [BS x num_classes]
            loss = loss_function(y,y_hat)
            selected = tf.squeeze(tf.random.uniform((bs,),0,10,dtype=tf.int64)) #sample from the output [BS x 1]
            y_hat = tf.gather(y_hat,selected,axis=1,batch_dims=1) #get the output for the selected class [BS x 1]
            output = tf.math.log(y_hat)#tf.math.log(output) #log the output [BS x 1]

        j = tape.jacobian(output,model.trainable_variables)
        layer_sizes = [tf.reduce_sum(tf.size(v)) for v in model.trainable_variables] #get the size of each layer
        j = [tf.reshape(j[i],(bs,layer_sizes[i])) for i in range(len(j))] #reshape the gradient to [BS x num_layer_params x layers]
        j = tf.concat(j,axis=1) #concat the gradient over the layers [BS x num_params]
        j = tf.square(j) #square the gradient [BS x num_params]
        j = tf.reduce_sum(j,axis=1) #sum the gradient [ 1]
        return j, y_hat,output
    
    @tf.function
    def Get_Z_sm_uniform_single(item,model):
        x,y = item
        x = tf.expand_dims(x,0)
        y = tf.expand_dims(y,0)
        bs = tf.shape(x)[0]
        loss_function = tf.keras.losses.CategoricalCrossentropy()
        
        with tf.GradientTape() as tape:
            #print(text_batch[i])
            y_hat = model(x,training=False) #get model output (softmax) [1 x num_classes]
            loss = loss_function(y,y_hat)
            selected = tf.random.uniform((bs,),0,10,dtype=tf.int64) #sample from the output [1 x 1]
            y_hat = tf.gather(y_hat,selected,axis=1,batch_dims=1) #get the output for the selected class [BS x 1]
            output = tf.math.log(y_hat)#tf.math.log(output) #log the output [BS x 1]

        j = tape.jacobian(output,model.trainable_variables)
        layer_sizes = [tf.reduce_sum(tf.size(v)) for v in model.trainable_variables] #get the size of each layer
        j = [tf.reshape(j[i],(bs,layer_sizes[i])) for i in range(len(j))] #reshape the gradient to [BS x num_layer_params x layers]
        j = tf.concat(j,axis=1) #concat the gradient over the layers [BS x num_params]
        j = tf.square(j) #square the gradient [BS x num_params]
        j = tf.reduce_sum(j,axis=1) #sum the gradient [ 1]
        return j, y_hat,output

    @tf.function
    def Get_Z_sm_sample_single(item,model):
        x,y = item
        x = tf.expand_dims(x,0)
        y = tf.expand_dims(y,0)
        bs = tf.shape(x)[0]
        loss_function = tf.keras.losses.CategoricalCrossentropy()
        
        with tf.GradientTape() as tape:
            #print(text_batch[i])
            y_hat = model(x,training=False) #get model output (softmax) [1 x num_classes]
            loss = loss_function(y,y_hat)
            selected = tf.random.uniform((bs,),0,10,dtype=tf.int64) #sample from the output [1 x 1]
            y_hat = tf.gather(y_hat,selected,axis=1,batch_dims=1) #get the output for the selected class [BS x 1]
            output = tf.math.log(y_hat)#tf.math.log(output) #log the output [BS x 1]

        j = tape.jacobian(output,model.trainable_variables)
        layer_sizes = [tf.reduce_sum(tf.size(v)) for v in model.trainable_variables] #get the size of each layer
        j = [tf.reshape(j[i],(bs,layer_sizes[i])) for i in range(len(j))] #reshape the gradient to [BS x num_layer_params x layers]
        j = tf.concat(j,axis=1) #concat the gradient over the layers [BS x num_params]
        j = tf.square(j) #square the gradient [BS x num_params]
        j = tf.reduce_sum(j,axis=1) #sum the gradient [ 1]
        return j, y_hat,output

    @tf.function
    def Get_Z_sm_sample_single_emp(item,model):
        x,y = item
        x = tf.expand_dims(x,0)
        y = tf.expand_dims(y,0)
        bs = tf.shape(x)[0]
        loss_function = tf.keras.losses.CategoricalCrossentropy()
        
        with tf.GradientTape() as tape:
            #print(text_batch[i])
            y_hat = model(x,training=False) #get model output (softmax) [1 x num_classes]
            loss = loss_function(y,y_hat)
            selected = tf.random.uniform((bs,),0,10,dtype=tf.int64) #sample from the output [1 x 1]
            #y_hat = tf.gather(y_hat,selected,axis=1,batch_dims=1) #get the output for the selected class [BS x 1]
            y_hat = tf.gather(y_hat,tf.argmax(y,axis=1),axis=1,batch_dims=1)
            output = tf.math.log(y_hat)#tf.math.log(output) #log the output [BS x 1]

        j = tape.jacobian(output,model.trainable_variables)
        layer_sizes = [tf.reduce_sum(tf.size(v)) for v in model.trainable_variables] #get the size of each layer
        j = [tf.reshape(j[i],(bs,layer_sizes[i])) for i in range(len(j))] #reshape the gradient to [BS x num_layer_params x layers]
        j = tf.concat(j,axis=1) #concat the gradient over the layers [BS x num_params]
        j = tf.square(j) #square the gradient [BS x num_params]
        j = tf.reduce_sum(j,axis=1) #sum the gradient [ 1]
        return j, y_hat,output
    
    def func(x,a):
        return a*np.log(1+x)**2
    
    @tf.function
    def tf_func(x,a):
        return a*tf.math.log(1+x)**2
    
    @tf.function
    def filter_func(x,y,alpha):
        j, y_hat,output = Get_Z_sm_sample_single_emp((x,y),model)
        print(tf_func(-output,alpha))
        print(j - tf_func(-output,alpha)>0)
        return tf.squeeze(j - tf_func(-output,alpha) > 0)

    # Train the model
    # epochs = 50
    # for epoch in range(epochs):
    #     print("\nStart of epoch %d" % (epoch,))
        
    #     #limit training dataset to only samples above the alpha curve
    #     sampleFIMs = [] #store the FIMs for each sample
    #     yHats = []
    #     for items in train_dataset.take(500):
    #         j, y_hat,output = Get_Z_sm_uniform_single(items,model)
    #         sampleFIMs.append(j)
    #         yHats.append(-output) #negative applied here
        
    #     sampleFIMs = tf.concat(sampleFIMs,axis=0)
    #     yHats = tf.concat(yHats,axis=0)
    #     popt, pcov = curve_fit(func, yHats.numpy(),sampleFIMs.numpy())
    #     alpha = popt[0]
    #     print("Alpha: ",alpha)
    #     #remove filtered dataset from memory
    #     if "filtered_train_dataset" in locals():
    #         del filtered_train_dataset
    #     filtered_train_dataset = train_dataset.filter(lambda x,y: filter_func(x,y,tf.cast(alpha,tf.float32))) #This is not optimal, but it will work for now
    #     filtered_train_dataset = filtered_train_dataset.shuffle(1000).batch(32)
    #     #filtered_train_dataset = filtered_train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    #     wandb.log({"Alpha":alpha,"MeanSampleFIM":tf.reduce_mean(sampleFIMs)},commit=False)

    #     #find the length of the filtered dataset
    #     count = 0
    #     for items in filtered_train_dataset:
    #         count += 1
    #     print("Filtered Dataset Length: ",count)
    #     wandb.log({"Filtered Dataset Length":count},commit=False)

    #     #save some examples of the filtered dataset
    #     # for items in filtered_train_dataset.take(10):
    #     #     x,y = items
    #     #     wandb.log({"Filtered Image":wandb.Image(x[0].numpy())})
    #     #     wandb.log({"Filtered Label":y[0].numpy()})

    #     #train for an epoch
    #     stats = model.fit(filtered_train_dataset, epochs=1, callbacks=[],validation_data=test_dataset)
    #     wandb.log({"loss":stats.history["loss"][0],
    #                 "accuracy":stats.history["accuracy"][0],
    #                 "val_loss":stats.history["val_loss"][0],
    #                 "val_accuracy":stats.history["val_accuracy"][0]},step=epoch+1,commit=True)

    train_DS = DataGen(x_train,y_train)
    for i in range(100):
        print(i)
        train_DS.on_epoch_begin()
        x,y = train_DS()
        print(x.shape)

        while train_DS()!=None:
            x,y = train_DS()
            #train
            stats = model.train_on_batch(x,y)
            #log
            wandb.log({"loss":stats[0],
                        "accuracy":stats[1]},commit=False)
        #train for an epoch
        stats = model.evaluate(test_dataset)
        wandb.log({"val_loss":stats[0],
                    "val_accuracy":stats[1]},commit=True)



        
def standard_train():
    @tf.function
    def Get_Z_sm_uniform(items,model):
        x,y = items
        bs = tf.shape(x)[0]
        loss_function = tf.keras.losses.CategoricalCrossentropy()
        
        with tf.GradientTape() as tape:
            #print(text_batch[i])
            y_hat = model(x,training=False) #get model output (softmax) [BS x num_classes]
            loss = loss_function(y,y_hat)
            selected = tf.squeeze(tf.random.uniform((bs,),0,10,dtype=tf.int64)) #sample from the output [BS x 1]
            y_hat = tf.gather(y_hat,selected,axis=1,batch_dims=1) #get the output for the selected class [BS x 1]
            output = tf.math.log(y_hat)#tf.math.log(output) #log the output [BS x 1]

        j = tape.jacobian(output,model.trainable_variables)
        layer_sizes = [tf.reduce_sum(tf.size(v)) for v in model.trainable_variables] #get the size of each layer
        j = [tf.reshape(j[i],(bs,layer_sizes[i])) for i in range(len(j))] #reshape the gradient to [BS x num_layer_params x layers]
        j = tf.concat(j,axis=1) #concat the gradient over the layers [BS x num_params]
        j = tf.square(j) #square the gradient [BS x num_params]
        j = tf.reduce_sum(j,axis=1) #sum the gradient [ 1]
        return j, y_hat,output
    
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


    train_dataset = train_dataset.map(map_fn).batch(32)
    test_dataset = test_dataset.map(map_fn)
    test_dataset = test_dataset.batch(32)

    # train_dataset = train_dataset.shuffle(50000).batch(32)
    # test_dataset = test_dataset.batch(32)
    # train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # test_dataset = test_dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

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


    # Compile the model
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    WandbCallback = wandb.keras.WandbCallback(save_model=False)

    def func(x,a):
        return a*np.log(1+x)**2
    epochs = 50
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        
        #limit training dataset to only samples above the alpha curve
        sampleFIMs = [] #store the FIMs for each sample
        yHats = []
        for items in train_dataset.take(500//32):
            j, y_hat,output = Get_Z_sm_uniform(items,model)
            sampleFIMs.append(j)
            yHats.append(-output) #negative applied here
        
        sampleFIMs = tf.concat(sampleFIMs,axis=0)
        yHats = tf.concat(yHats,axis=0)
        popt, pcov = curve_fit(func, yHats.numpy(),sampleFIMs.numpy())
        alpha = popt[0]
        print("Alpha: ",alpha)
        wandb.log({"Alpha":alpha,"MeanSampleFIM":tf.reduce_mean(sampleFIMs)},commit=False)

        #find the length of the filtered dataset
        count = 0
        for items in train_dataset:
            count += 1
        print("Filtered Dataset Length: ",count)
        wandb.log({"Filtered Dataset Length":count},commit=False)

        #save some examples of the filtered dataset
        # for items in filtered_train_dataset.take(10):
        #     x,y = items
        #     wandb.log({"Filtered Image":wandb.Image(x[0].numpy())})
        #     wandb.log({"Filtered Label":y[0].numpy()})

        #train for an epoch
        stats = model.fit(train_dataset, epochs=1, callbacks=[],validation_data=test_dataset)
        wandb.log({"loss":stats.history["loss"][0],
                    "accuracy":stats.history["accuracy"][0],
                    "val_loss":stats.history["val_loss"][0],
                    "val_accuracy":stats.history["val_accuracy"][0]},step=epoch+1,commit=True)

def alpha_weight(name="standard",hard_type="train"):
    # Load the data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train/255
    x_test = x_test/255

    #creata hard subset from 
    if hard_type == "train":
        (x_c100_train, y_c100_train), (x_c100_test, y_c100_test) = tf.keras.datasets.cifar100.load_data(label_mode="coarse")

        #/255
        x_c100_train = x_c100_train/255
        x_c100_test = x_c100_test/255
        
        #limit to only superclass 5
        x_c100_train_limited = [x for x,y in zip(x_c100_train,y_c100_train) if y == [5]]
        y_c100_train_limited = [y for x,y in zip(x_c100_train,y_c100_train) if y == [5]]
        x_c100_test_limited = [x for x,y in zip(x_c100_test,y_c100_test) if y == [5]]
        y_c100_test_limited = [y for x,y in zip(x_c100_test,y_c100_test) if y == [5]]

        x_c100_train = x_c100_train_limited
        y_c100_train = y_c100_train_limited
        x_c100_test = x_c100_test_limited
        y_c100_test = y_c100_test_limited

        hard_test_dataset = tf.data.Dataset.from_tensor_slices((x_c100_test, y_c100_test))

        #add the class to the cifar10 dataset
        x_train = np.concatenate((x_train,x_c100_train),axis=0)
        y_train = np.concatenate((y_train,y_c100_train),axis=0)
        x_test = np.concatenate((x_test,x_c100_test),axis=0)
        y_test = np.concatenate((y_test,y_c100_test),axis=0)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)) 
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    def map_fn(image, label):
        #image = tf.cast(image, tf.float32)
        #image = tf.expand_dims(image, -1)
        if hard_type == "train":
            return image, tf.squeeze(tf.one_hot(tf.cast(label,tf.int32), 11, on_value=1.0))
        else:
            return image, tf.squeeze(tf.one_hot(tf.cast(label,tf.int32), 10, on_value=1.0))


    train_dataset = train_dataset.map(map_fn)
    test_dataset = test_dataset.map(map_fn)

    train_dataset = train_dataset.shuffle(50000).batch(32)
    test_dataset = test_dataset.batch(32)
    train_dataset = train_dataset.cache()
    test_dataset = test_dataset.cache()

    if hard_type == "train":
        hard_test_dataset = hard_test_dataset.map(map_fn)
        hard_test_dataset = hard_test_dataset.batch(32)

    # Load the model
    class Model(tf.keras.Model):
        def __init__(self,train_type):
            super(Model,self).__init__()
            self.train_type = train_type
            self.conv1 = tf.keras.layers.Conv2D(32, (3,3), activation='relu')
            self.MP1 = tf.keras.layers.MaxPooling2D((2,2))
            self.conv2 = tf.keras.layers.Conv2D(64, (3,3), activation='relu')
            self.MP2 = tf.keras.layers.MaxPooling2D((2,2))
            self.conv3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu')
            self.flatten = tf.keras.layers.Flatten()
            self.dense1 = tf.keras.layers.Dense(64, activation='relu')
            if hard_type == "train":
                self.dense2 = tf.keras.layers.Dense(11, activation='softmax')
            else:
                self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

            self.loss_function = tf.keras.losses.CategoricalCrossentropy()
            self.nored_loss_function = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

            #self.batch_count = tf.Variable(0)
            # self.grad_scale_1 = tf.Variable(0)
            # self.grad_scale_2 = tf.Variable(0)
            # self.total_batches = tf.constant(1641)

        def call(self,x):
            x = self.conv1(x)
            x = self.MP1(x)
            x = self.conv2(x)
            x = self.MP2(x)
            x = self.conv3(x)
            x = self.flatten(x)
            x = self.dense1(x)
            x = self.dense2(x)
            return x
            
        @tf.function
        def train_step(self,data):
            if self.train_type == "standard":
                return self.standard_train_step(data)
            elif self.train_type == "f_norm":
                return self.F_norm_train_step(data)
            elif self.train_type == "batch_loss_scale":
                return self.batch_lossscale_gradient_step(data)
            elif self.train_type == "batch_y_scale":
                return self.batch_yscale_gradient_step(data)
            elif self.train_type == "batch_nonlog_y_scale":
                return self.batch_nonlog_yscale_gradient_step(data)
            elif self.train_type == "batch_nonlog_inverse_y_scale":
                return self.batch_nonlog_inverse_yscale_gradient_step(data)
            elif self.train_type == "individual_FIM_scale":
                return self.individual_FIM_gradient_step(data)
            elif self.train_type == "individual_y_scale":
                return self.individual_yscale_gradient_step(data)
            elif self.train_type == "fixed_scale":
                return self.batch_fixed_scale_gradient_step(data)
            elif self.train_type == "fixed_y_scale":
                return self.batch_fixed_y_scale_gradient_step(data)
            elif self.train_type == "fixed_alpha_scale":
                return self.batch_fixed_alpha_scale_gradient_step(data)
            elif self.train_type == "max_alpha_scale":
                return self.max_fixed_alpha_scale_gradient_step(data)
            elif self.train_type == "max_y_scale":
                return self.max_fixed_y_scale_gradient_step(data)
            
            else:
                print("Not a valid training type")

        @tf.function
        def test_step(self, data):
            # Unpack the data
            x, y = data
            # Compute predictions
            y_pred = self(x, training=False)
            # Updates the metrics tracking the loss
            self.compute_loss(y=y, y_pred=y_pred)
            # Update the metrics.
            for metric in self.metrics:
                if metric.name != "loss":
                    metric.update_state(y, y_pred)
            # Return a dict mapping metric names to current value.
            # Note that it will include the loss (tracked in self.metrics).
            return {m.name: m.result() for m in self.metrics}
        
        @tf.function
        def F_norm_train_step(self,data):
            x,y = data
            with tf.GradientTape() as tape:
                y_hat = self(x,training=True)
                loss = self.compiled_loss(y,y_hat)
            batch_grads = tape.jacobian(loss,self.trainable_variables) # [batch_size x num_layers x num_params]
            batch_F = [[tf.square(l) for l in g] for g in batch_grads]
            max_F = max([tf.reduce_max(tf.reduce_max(f,axis=0)) for f in batch_F])
            #normalise F
            batch_F = [[f/max_F for f in g] for g in batch_F]
            
            #scale the grad by the FIM
            batch_grads = [[g*tf.cast(f,tf.float32) for g,f in zip(g,b)] for g,b in zip(batch_grads,batch_F)]
            #sum the grads over the batch
            grad = [tf.reduce_sum(g,axis=0) for g in batch_grads]

            #calc the gradient scale
            grad_scale = tf.reduce_mean([tf.reduce_mean(tf.abs(g)) for g in grads])
            wandb.log({"grad_scale":grad_scale},commit=False)
            
            self.optimizer.apply_gradients(zip(grad,self.trainable_variables))
            self.compiled_metrics.update_state(y,y_hat)
            return {m.name:m.result() for m in self.metrics}

        @tf.function
        def batch_lossscale_gradient_step(self,data):
            x,y = data
            with tf.GradientTape() as tape:
                y_hat = self(x,training=True)
                loss = self.compiled_loss(y,y_hat)
            grads = tape.gradient(loss,self.trainable_variables)
            #scale the grad by the loss
            grads = [g*loss for g in grads]
            #calc the gradient scale
            grad_scale = tf.reduce_mean([tf.reduce_mean(tf.abs(g)) for g in grads])

            self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
            self.compiled_metrics.update_state(y,y_hat)
            results = {m.name:m.result() for m in self.metrics}
            results["grad_scale"] = grad_scale
            return results
        
        def individual_yscale_gradient_step(self,data):
            x,y = data
            with tf.GradientTape() as tape:
                y_hat = self(x,training=True) #[batch_size x num_classes]
                nored_loss = self.nored_loss_function(y,y_hat) #[BS x 1]
                loss = self.compiled_loss(y,y_hat) #[ 1]
            batch_grads = tape.jacobian(nored_loss,self.trainable_variables) #[layers x (batch_size, num_params)]
            #calc the y_hat for correct output
            y_hat_temp = tf.gather(y_hat,tf.argmax(y,axis=1),axis=1,batch_dims=1) # [bs x 1]
            #log the y_hat
            y_hat_temp = -tf.math.log(y_hat_temp) # [BS x 1]

            layer_ranks = [(-1,1,1,1,1),
                            (-1,1),
                            (-1,1,1,1,1),
                            (-1,1),
                            (-1,1,1,1,1),
                            (-1,1),
                            (-1,1,1),
                            (-1,1),
                            (-1,1,1),
                            (-1,1)]
            y_hat_temp = [tf.reshape(y_hat_temp,lr) for lr in layer_ranks]
            batch_grads = [g*y for g,y in zip(batch_grads,y_hat_temp)]

            #mean over the batch
            grads = [tf.reduce_mean(g,axis=0) for g in batch_grads]

            grad_scale = tf.reduce_sum([tf.reduce_mean(tf.abs(g)) for g in grads])

            self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
            self.compiled_metrics.update_state(y,y_hat)
            results =  {m.name:m.result() for m in self.metrics}
            results["grad_scale"] = grad_scale
            return results

        @tf.function
        def batch_yscale_gradient_step(self,data):
            x,y = data
            with tf.GradientTape() as tape:
                y_hat = self(x,training=True) #[batch_size x num_classes]
                loss = self.compiled_loss(y,y_hat) #[ 1]

            grads = tape.gradient(loss,self.trainable_variables) #[layers x (batch_size, num_params)]
            #calc the y_hat for correct output
            y_hat_temp = tf.reduce_mean(tf.gather(y_hat,tf.argmax(y,axis=1),axis=1,batch_dims=1)) # [1]
            #log the y_hat
            y_hat_temp = -tf.math.log(y_hat_temp) # [BS x 1]

            grads = [g_l*y_hat_temp for g_l in grads]
            grad_scale = tf.reduce_sum([tf.reduce_mean(tf.abs(g)) for g in grads])

            self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
            self.compiled_metrics.update_state(y,y_hat)
            results =  {m.name:m.result() for m in self.metrics}
            results["grad_scale"] = grad_scale
            #multiply the grad_scale by the learning rate
            #check id decayed lr exists
            if hasattr(self.optimizer,"_decayed_lr"):
                #if it does, multiply by the decayed lr
                lr = self.optimizer._decayed_lr(tf.float32)
                rel_grad_scale = grad_scale * lr
            else:
                #if it doesn't, multiply by the learning rate
                lr = self.optimizer.learning_rate
                rel_grad_scale = grad_scale * lr
            results["lr"] = lr
            results["rel_grad_scale"] = rel_grad_scale
            return results
        
        def batch_nonlog_yscale_gradient_step(self,data):
            #fails to train with lr of 0.01
            x,y = data
            with tf.GradientTape() as tape:
                y_hat = self(x,training=True) #[batch_size x num_classes]
                loss = self.compiled_loss(y,y_hat) #[ 1]

            grads = tape.gradient(loss,self.trainable_variables) #[layers x (batch_size, num_params)]
            #calc the y_hat for correct output
            y_hat_temp = tf.reduce_mean(tf.gather(y_hat,tf.argmax(y,axis=1),axis=1,batch_dims=1)) # [1]
            #log the y_hat
            y_hat_temp = -y_hat_temp # [BS x 1]

            grads = [g_l*y_hat_temp for g_l in grads]
            grad_scale = tf.reduce_sum([tf.reduce_mean(tf.abs(g)) for g in grads])

            self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
            self.compiled_metrics.update_state(y,y_hat)
            results =  {m.name:m.result() for m in self.metrics}
            results["grad_scale"] = grad_scale
            return results
        
        def batch_nonlog_inverse_yscale_gradient_step(self,data):
            x,y = data
            with tf.GradientTape() as tape:
                y_hat = self(x,training=True) #[batch_size x num_classes]
                loss = self.compiled_loss(y,y_hat) #[ 1]

            grads = tape.gradient(loss,self.trainable_variables) #[layers x (batch_size, num_params)]
            #calc the y_hat for correct output
            y_hat_temp = 1- tf.reduce_mean(tf.gather(y_hat,tf.argmax(y,axis=1),axis=1,batch_dims=1)) # [1]

            grads = [g_l*y_hat_temp for g_l in grads]
            grad_scale = tf.reduce_sum([tf.reduce_mean(tf.abs(g)) for g in grads])

            self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
            self.compiled_metrics.update_state(y,y_hat)
            results =  {m.name:m.result() for m in self.metrics}
            results["grad_scale"] = grad_scale
            return results
        
        @tf.function
        def individual_FIM_gradient_step(self,data):
            x,y = data
            #calc batch FIM
            with tf.GradientTape() as tape:
                y_hat = self(x,training=True)
                loss = self.compiled_loss(y,y_hat)
                output = tf.gather(y_hat,tf.argmax(y,axis=1),axis=1,batch_dims=1) #get the output for the selected class [BS x 1]
                output = tf.math.log(output)

            #get the mean F for the batch
            batch_grads = tape.jacobian(output,self.trainable_variables) # [batch_size x num_layers x num_params]
            #get the layer shapes
            layer_sizes = [tf.range(1,tf.rank(v)+1) for v in self.trainable_variables]
            #calc the FIM
            batch_F = [tf.square(g) for g in batch_grads] #square each layer [num_layers x (batch_size, num_params)]
            print(batch_F)
            batch_F = [tf.reduce_sum(g,axis=ls) for g,ls in zip(batch_F,layer_sizes)] #sum over each layer [num_layers x (batch_size)]
            batch_F = tf.reduce_sum(batch_F,axis=0) #sum over the layers [batch_size]
            print(batch_F)
            F = tf.reduce_mean(batch_F,axis=0) #mean over the batch []

            #noramlise the batch grads by the FIM
            batch_grads = [l/ F for l in batch_grads]

            #mean over the batch
            grads = [tf.reduce_mean(g,axis=0) for g in batch_grads]

            grad_scale = tf.reduce_sum([tf.reduce_mean(tf.abs(g)) for g in grads])

            self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
            self.compiled_metrics.update_state(y,y_hat)
            results =  {m.name:m.result() for m in self.metrics}
            results["grad_scale"] = grad_scale
            return results

        @tf.function
        def batch_fixed_scale_gradient_step(self,data):
            x,y = data
            with tf.GradientTape() as tape:
                y_hat = self(x,training=True)
                loss = self.compiled_loss(y,y_hat)
            
            #make the gradient have a scale of 0.3
            grads = tape.gradient(loss,self.trainable_variables)
            grad_scale = tf.reduce_mean([tf.reduce_mean(tf.abs(g)) for g in grads])
            grads = [g*0.3/grad_scale for g in grads]
            #calc the gradient scale
            grad_scale = tf.reduce_mean([tf.reduce_mean(tf.abs(g)) for g in grads])
            
            self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
            self.compiled_metrics.update_state(y,y_hat)
            results = {m.name:m.result() for m in self.metrics}
            results["grad_scale"] = grad_scale
            if hasattr(self.optimizer,"_decayed_lr"):
                #if it does, multiply by the decayed lr
                lr = self.optimizer._decayed_lr(tf.float32)
                rel_grad_scale = grad_scale * lr
            else:
                #if it doesn't, multiply by the learning rate
                lr = self.optimizer.learning_rate
                rel_grad_scale = grad_scale * lr
            results["lr"] = lr
            results["rel_grad_scale"] = rel_grad_scale
            return results
        
        @tf.function
        def batch_fixed_y_scale_gradient_step(self,data):
            x,y = data
            with tf.GradientTape() as tape:
                y_hat = self(x,training=True)
                loss = self.compiled_loss(y,y_hat)
            
            #make the gradient have a scale of 0.3
            grads = tape.gradient(loss,self.trainable_variables)
            y_hat_temp = tf.reduce_mean(tf.gather(y_hat,tf.argmax(y,axis=1),axis=1,batch_dims=1)) # [1]
            #log the y_hat
            y_hat_temp = -tf.math.log(y_hat_temp) # [BS x 1]

            grad_scale = tf.reduce_mean([tf.reduce_mean(tf.abs(g)) for g in grads])
            k=0.3
            grads = [g*k/grad_scale for g in grads] 
            grads = [g_l*y_hat_temp for g_l in grads]
            #calc the gradient scale
            grad_scale = tf.reduce_mean([tf.reduce_mean(tf.abs(g)) for g in grads])
            
            self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
            self.compiled_metrics.update_state(y,y_hat)
            results = {m.name:m.result() for m in self.metrics}
            results["grad_scale"] = grad_scale
            if hasattr(self.optimizer,"_decayed_lr"):
                #if it does, multiply by the decayed lr
                lr = self.optimizer._decayed_lr(tf.float32)
                rel_grad_scale = grad_scale * lr
            else:
                #if it doesn't, multiply by the learning rate
                lr = self.optimizer.learning_rate
                rel_grad_scale = grad_scale * lr
            results["lr"] = lr
            results["rel_grad_scale"] = rel_grad_scale
            return results

        @tf.function
        def max_fixed_y_scale_gradient_step(self,data):
            x,y = data
            with tf.GradientTape() as tape:
                y_hat = self(x,training=True)
                loss = self.compiled_loss(y,y_hat)
            
            #make the gradient have a scale of 0.3
            grads = tape.gradient(loss,self.trainable_variables)
            y_hat_temp = tf.reduce_max(tf.gather(y_hat,tf.argmax(y,axis=1),axis=1,batch_dims=1)) # [1]
            #log the y_hat
            y_hat_temp = -tf.math.log(y_hat_temp) # [BS x 1]

            grads = [g_l*y_hat_temp for g_l in grads]
            #calc the gradient scale
            grad_scale = tf.reduce_mean([tf.reduce_mean(tf.abs(g)) for g in grads])
            
            self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
            self.compiled_metrics.update_state(y,y_hat)
            results = {m.name:m.result() for m in self.metrics}
            results["grad_scale"] = grad_scale
            if hasattr(self.optimizer,"_decayed_lr"):
                #if it does, multiply by the decayed lr
                lr = self.optimizer._decayed_lr(tf.float32)
                rel_grad_scale = grad_scale * lr
            else:
                #if it doesn't, multiply by the learning rate
                lr = self.optimizer.learning_rate
                rel_grad_scale = grad_scale * lr
            results["lr"] = lr
            results["rel_grad_scale"] = rel_grad_scale
            return results

                
        @tf.function
        def batch_fixed_alpha_scale_gradient_step(self,data):
            x,y = data
            with tf.GradientTape() as tape:
                y_hat = self(x,training=True)
                loss = self.compiled_loss(y,y_hat)
            
            #make the gradient have a scale of 0.3
            grads = tape.gradient(loss,self.trainable_variables)
            y_hat_temp = tf.reduce_mean(tf.gather(y_hat,tf.argmax(y,axis=1),axis=1,batch_dims=1)) # [1]
            #log the y_hat
            F_proxy = tf.math.log(1-tf.math.log(y_hat_temp))**2 # [BS x 1]

            grad_scale = tf.reduce_mean([tf.reduce_mean(tf.abs(g)) for g in grads])
            k=0.5
            grads = [g_l*k*F_proxy for g_l in grads]
            #calc the gradient scale
            grad_scale = tf.reduce_mean([tf.reduce_mean(tf.abs(g)) for g in grads])
            
            self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
            self.compiled_metrics.update_state(y,y_hat)
            results = {m.name:m.result() for m in self.metrics}
            results["grad_scale"] = grad_scale
            if hasattr(self.optimizer,"_decayed_lr"):
                #if it does, multiply by the decayed lr
                lr = self.optimizer._decayed_lr(tf.float32)
                rel_grad_scale = grad_scale * lr
            else:
                #if it doesn't, multiply by the learning rate
                lr = self.optimizer.learning_rate
                rel_grad_scale = grad_scale * lr
            results["lr"] = lr
            results["rel_grad_scale"] = rel_grad_scale
            return results
        
        @tf.function
        def max_fixed_alpha_scale_gradient_step(self,data):
            x,y = data
            with tf.GradientTape() as tape:
                y_hat = self(x,training=True)
                loss = self.compiled_loss(y,y_hat)
            
            #make the gradient have a scale of 0.3
            grads = tape.gradient(loss,self.trainable_variables)
            y_hat_temp = tf.reduce_min(tf.gather(y_hat,tf.argmax(y,axis=1),axis=1,batch_dims=1)) # [1]
            #log the y_hat
            F_proxy = tf.math.log(1-tf.math.log(y_hat_temp))**2 # [BS x 1]

            grad_scale = tf.reduce_mean([tf.reduce_mean(tf.abs(g)) for g in grads])
            k=0.5
            grads = [g_l*k*F_proxy for g_l in grads]
            #calc the gradient scale
            grad_scale = tf.reduce_mean([tf.reduce_mean(tf.abs(g)) for g in grads])
            
            self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
            self.compiled_metrics.update_state(y,y_hat)
            results = {m.name:m.result() for m in self.metrics}
            results["grad_scale"] = grad_scale
            if hasattr(self.optimizer,"_decayed_lr"):
                #if it does, multiply by the decayed lr
                lr = self.optimizer._decayed_lr(tf.float32)
                rel_grad_scale = grad_scale * lr
            else:
                #if it doesn't, multiply by the learning rate
                lr = self.optimizer.learning_rate
                rel_grad_scale = grad_scale * lr
            results["lr"] = lr
            results["rel_grad_scale"] = rel_grad_scale
            return results

        def add_grad_scale(self,grad_scale):
            #tf.print("Batch count: ",self.optimizer.batch_count," Total batches: ",self.optimizer.total_batches)
            if self.optimizer.batch_count < self.optimizer.total_batches//2:
                #if we are in the first half of the training, add the grad_scale to the grad_scale_1
                self.optimizer.grad_scale_1.assign_add(grad_scale)

            else:
                #if we are in the second half of the training, add the grad_scale to the grad_scale_2
                self.optimizer.grad_scale_2.assign_add(grad_scale)
            self.optimizer.batch_count.assign_add(1)



        @tf.function
        def standard_train_step(self,data):
            
            x,y = data
            with tf.GradientTape() as tape:
                y_hat = self(x,training=True)
                loss = self.compiled_loss(y,y_hat)
            grads = tape.gradient(loss,self.trainable_variables)
            grad_scale = tf.reduce_sum([tf.reduce_mean(tf.abs(g)) for g in grads])
            self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
            self.compiled_metrics.update_state(y,y_hat)
            results =  {m.name:m.result() for m in self.metrics}
            results["grad_scale"] = grad_scale
            #self.add_grad_scale(grad_scale)
            
            #multiply the grad_scale by the learning rate
            #check id decayed lr exists
            #print the attributes of the optimizer
            if hasattr(self.optimizer,"_decayed_lr"):
                #if it does, multiply by the decayed lr
                lr = self.optimizer._decayed_lr(tf.float32)
                rel_grad_scale = grad_scale * lr
            else:
                #if it doesn't, multiply by the learning rate
                lr = self.optimizer.learning_rate
                rel_grad_scale = grad_scale * lr
            results["lr"] = lr
            results["rel_grad_scale"] = rel_grad_scale
            
            return results
    
    class CustomCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super(CustomCallback, self).__init__()
            pass
        
        def on_epoch_end(self, epoch, logs=None):
            #save metrics to wandb
            #get most recent logs
            print(logs)
            wandb.log(logs,commit=True)

    class AdditionalValidationSets(tf.keras.callbacks.Callback):
        def __init__(self, validation_sets, verbose=0, batch_size=None):
            """
            :param validation_sets:
            2-tuples (validation_data, validation_set_name)
            :param verbose:
            verbosity mode, 1 or 0
            :param batch_size:
            batch size to be used when evaluating on the additional datasets
            """
            super(AdditionalValidationSets, self).__init__()
            self.validation_sets = validation_sets
            for validation_set in self.validation_sets:
                if len(validation_set) != 2:
                    raise ValueError()
            self.verbose = verbose
            self.batch_size = batch_size

        def on_epoch_end(self, epoch, logs=None):

            # evaluate on the additional validation sets
            for validation_set in self.validation_sets:
                if len(validation_set) == 2:
                    validation_data, validation_set_name = validation_set
                else:
                    raise ValueError()

                results = self.model.evaluate(validation_data,
                                            verbose=self.verbose,
                                            batch_size=self.batch_size)

                for metric, result in zip(self.model.metrics_names,results):
                    valuename = validation_set_name + '_' + metric
                    wandb.log({valuename: result}, commit=False)
        
    class GSLRscheduleCB(tf.keras.callbacks.Callback):
        def __init__(self, k ):
            super(GSLRscheduleCB, self).__init__()
            #decay the learning rate by k if the average gradient scale decreased last epoch
            self.k = k
            self.prev_grad_scale = None

        def on_epoch_end(self, epoch, logs=None):
            #update at the end of each epoch
            if epoch == 0:
                #get the average gradient scale from the logs
                self.prev_grad_scale = logs["grad_scale"]
                tf.print("First epoch",self.prev_grad_scale)
            else:
                if logs["grad_scale"] < self.prev_grad_scale:
                    #if the gradient scale decreased, decay the learning rate
                    self.model.optimizer.learning_rate = self.model.optimizer.learning_rate * self.k
                self.prev_grad_scale = logs["grad_scale"]

    # class GSLRscheduleCBV2(tf.keras.callbacks.Callback):
    #     def __init__(self, k ):
    #         super(GSLRscheduleCBV2, self).__init__()
    #         #decay the learning rate by k if the average gradient scale decreased last epoch
    #         #this looks within the epoch to calc grad scale change
    #         self.k = k

    #     def on_epoch_end(self, epoch, logs=None):
    #         #update at the end of each epoch
    #         first_half = self.model.optimizer.grad_scale_1 / (self.model.optimizer.total_batches//2)
    #         second_half = self.model.optimizer.grad_scale_2 / (self.model.optimizer.total_batches//2)
    #         tf.print("First half: ",first_half," Second half: ",second_half)
    #         if first_half > second_half:
    #             #if the gradient scale decreased, decay the learning rate
    #             self.model.optimizer.learning_rate = self.model.optimizer.learning_rate * self.k
    #             #reset the grad scales
    #         self.model.optimizer.grad_scale_1.assign(0.0)
    #         self.model.optimizer.grad_scale_2.assign(0.0)
    #         self.model.optimizer.batch_count.assign(0)
        
            

    
    class GSLRscheduleCBV3(tf.keras.callbacks.Callback):
        def __init__(self, k ,LB):
            super(GSLRscheduleCBV3, self).__init__()
            #decay the learning rate by k if the average gradient scale decreased last epoch
            self.k =k
            # self.a = a
            # self.b = b
            self.LB = LB
            self.prev_grad_scale = []

        def on_epoch_end(self, epoch, logs=None):
            #update at the end of each epoch
            if len(self.prev_grad_scale) == 0:
                #get the average gradient scale from the logs
                self.prev_grad_scale.append(logs["grad_scale"])
                tf.print("First epoch",self.prev_grad_scale)
            else:
                self.prev_grad_scale.append(logs["grad_scale"])
                #calc the average direction change over the last LB epochs
                diffs = [self.prev_grad_scale[i] - self.prev_grad_scale[i-1] for i in range(1,len(self.prev_grad_scale))]
                avg_diff = tf.reduce_mean(diffs)

                #remove the oldest diff
                if len(self.prev_grad_scale) > self.LB:
                    self.prev_grad_scale.pop(0)

                if avg_diff < 0:
                    #if the gradient scale decreased, decay the learning rate
                    self.model.optimizer.learning_rate = self.model.optimizer.learning_rate * self.k
                else:
                    #if the gradient scale increased, increase the learning rate
                    self.model.optimizer.learning_rate = self.model.optimizer.learning_rate / self.k
                #if the gradient scale decreased, decay the learning rate
                
    class GSLRscheduleCBV4(tf.keras.callbacks.Callback):
        def __init__(self, a=50, b=1, extent=0.1, LB=3):
            super(GSLRscheduleCBV4, self).__init__()
            #decay the learning rate by k if the average gradient scale decreased last epoch
            self.extent = extent
            self.a = a
            self.b = b
            self.LB = LB
            self.prev_grad_scale = []

        def on_epoch_end(self, epoch, logs=None):
            #update at the end of each epoch
            if len(self.prev_grad_scale) == 0:
                #get the average gradient scale from the logs
                self.prev_grad_scale.append(logs["grad_scale"])
                tf.print("First epoch",self.prev_grad_scale)
            else:
                self.prev_grad_scale.append(logs["grad_scale"])
                #calc the average direction change over the last LB epochs
                diffs = [self.prev_grad_scale[i] - self.prev_grad_scale[i-1] for i in range(1,len(self.prev_grad_scale))]
                avg_diff = tf.reduce_mean(diffs)

                #remove the oldest diff
                if len(self.prev_grad_scale) > self.LB:
                    self.prev_grad_scale.pop(0)
                if avg_diff < 0 :
                    #if the gradient scale decreased, decay the learning rate
                    self.model.optimizer.learning_rate = self.model.optimizer.learning_rate * (((2*self.extent*self.b)/(self.b + np.exp(-self.a*avg_diff))) + (1-self.extent))

                

    class GSLRschedule(tf.keras.optimizers.SGD):
        def __init__(self, initial_learning_rate=0.01):
            super(GSLRschedule, self).__init__()
            self.learning_rate = initial_learning_rate
            self.grad_scale_1 = tf.Variable(0.0)
            self.grad_scale_2 = tf.Variable(0.0)
            self.total_batches = 1641
            self.batch_count = tf.Variable(0)


        def __call__(self, step):
            return self.learning_rate

    # Compile the model
    model = Model(name)
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=0.01,
    #     decay_steps=80000,
    #     decay_rate=0.1)
    # lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    #     initial_learning_rate=0.01,
    #     decay_steps=80000,
    #     alpha=0.1)
    
    GSLR = GSLRscheduleCBV4(a=50,b=1,extent=0.5,LB=3)
    lr_schedule = 0.01
    #optimizer = GSLRschedule(initial_learning_rate=0.01)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    cc = CustomCallback()
    if hard_type == "train":
        extra_test_eval = AdditionalValidationSets([(hard_test_dataset,"hard_test")],verbose=1,batch_size=32)

        #we want to weight samples more heavily if they are above the curve
        stats = model.fit(train_dataset, epochs=200, callbacks=[extra_test_eval,cc,GSLR],validation_data=test_dataset)
    else:
        stats = model.fit(train_dataset, epochs=200, callbacks=[cc],validation_data=test_dataset)
                
if __name__ == "__main__":
    os.environ['WANDB_API_KEY'] = 'fc2ea89618ca0e1b85a71faee35950a78dd59744'
    wandb.login()
    name = "standard"
    descriptor = "ClipGSLRV4_50_1_0.5_3"
    hard_type = "train"
    wandb.init(project="AlphaScaling",name=name+"_"+descriptor,config={"name":name,"descriptor":descriptor,"hard_type":hard_type})

    #main()
    alpha_weight(name=name,hard_type=hard_type)
    #standard_train()
    wandb.finish()