#calculate the Fisher infomration matrix for a regression model

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#set the random seed
np.random.seed(0)

x = np.random.normal(0,1,100)*10
y = 0.2*x**3 + 2*x**2 + (np.random.normal(0,1,100)-0.5)*150

#plot the data
plt.scatter(x,y)
#save the figure
plt.savefig('data.png')

#build the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1,input_shape=(1,)))
#model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(64,activation='relu'))
model.add(tf.keras.layers.Dense(1))

model.summary()
#compile the model
model.compile(optimizer='adam',loss='mse')

def fisher(model,x_f,y_f):
    total_f = 0
    for x,y in zip(x_f,y_f):
        #expand the dimensions
        x = tf.expand_dims(x,axis=0)
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            #calc the hessian of the log likelihood
            y_pred = model(x)
            loss = tf.keras.losses.MSE(y_pred,y)

        #calc the gradient of the loss with respect to the model parameters
        grads = tape.gradient(loss,model.trainable_variables)
        grads = [tf.reshape(grad,[-1]) for grad in grads]
        grads = tf.concat(grads,axis=0)
        #take the tace of the gTg matrix without multiplying the off diagonal elements
        grads = tf.square(grads)
        #sum the grads
        f = tf.reduce_sum(grads,axis=0)

        total_f += f
    return f/len(x_f)

def new_fisher(model,x_f,y_f):
    total_f = 0

    residuals = []
    for x,y in zip(x_f,y_f):
        #data losses
        x = tf.expand_dims(x,axis=0)
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            y_pred = model(x)
        residuals.append(y - y_pred)
    
    #calc the log likelihood from the residuals
    residuals = tf.concat(residuals,axis=0)
    std_dev = tf.math.reduce_std(residuals)
    log_likelihood = -tf.math.log(std_dev) - 0.5*tf.math.reduce_sum(tf.square(residuals))/std_dev**2

    #calc the hessian of the log likelihood
    with tf.GradientTape() as t2:
        with tf.GradientTape() as t1:
            tape.watch(model.trainable_variables)
 
        g = t1.gradient(log_likelihood, model.trainable_variables)
        print(g)

    H = t2.jacobian(g, model.trainable_variables)
    #take the tace of the hessian
    trace_H = sum(tf.linalg.trace(h) for h in H)
    print(trace_H)
    pnt()
    return trace_H

    



def reduced_dataset(model,x,percentage=10,high=False):
    #return the x values that are lowest loss
    y = model(x)
    loss = tf.keras.losses.MSE(y,x)
    loss = loss.numpy()
    loss = np.argsort(loss)

    if high:
        #take highest 10% of the dataset
        loss = loss[int(-len(loss)*(percentage/100)):]
    else:
        #take lowest 10% of the dataset
        loss = loss[:int(len(loss)*(percentage/100))]

    return np.array(x[loss])


#train the model
total_epochs = 5000
evaluation_points = 50
colors = cm.rainbow(np.linspace(0, 1, evaluation_points))

f = []
f_low = []
f_high = []
for i in range(evaluation_points):
    print(i)
    model.fit(x,y,epochs=total_epochs//evaluation_points,batch_size=5,verbose=0)

    #print the predicted values on the graph
    plt.scatter(x,model.predict(x),color=colors[i])

    f.append(new_fisher(model,x,y).numpy())
    x_low = reduced_dataset(model,x,percentage=50,high=False)
    f_low.append(fisher(model,x_low).numpy())
    x_high = reduced_dataset(model,x,percentage=50,high=True)
    f_high.append(fisher(model,x_high).numpy())

plt.savefig('fit.png')
plt.clf()

print(f)
print(f_low)
print(f_high)

plt.scatter(range(evaluation_points),f_low,color='red',label='Lowest 10%')
plt.scatter(range(evaluation_points),f_high,color='blue',label='Highest 10%')
plt.scatter(range(evaluation_points),f,color='black',label='All Data')
plt.legend()
plt.yscale('log')
plt.savefig('fisher.png')



