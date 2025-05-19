import atexit
import signal
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import pandas as pd
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize to [0, 1]
y_train, y_test = tf.keras.utils.to_categorical(y_train, 10), tf.keras.utils.to_categorical(y_test, 10)

# Define a 5-layer CNN model
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Function to compute the Fisher Information Matrix (FIM)
def compute_fim(model, x_all, y_all, FIM_type="stat"):
    
    @tf.function
    def fim_func_stat(model, x, y):
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            y_pred = model(x, training=False)
            #select output based on the output distribution
            selected = tf.squeeze(tf.random.categorical(tf.math.log(y_pred), 1)) #sample from the output [BS x 1]
            output = tf.gather(y_pred,selected,axis=1)
            grads = tape.gradient(tf.math.log(output), model.trainable_variables)
        summed = [tf.reduce_sum(tf.square(g)) for g in grads]
        return tf.reduce_sum(tf.where(tf.math.is_nan(summed), tf.zeros_like(summed), summed))

    @tf.function
    def fim_func_emp(model, x, y):
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            y_pred = model(x, training=False)
            #select the label classes
            output = tf.gather(y_pred, tf.argmax(y, axis=1), axis=1)
            grads = tape.gradient(tf.math.log(output), model.trainable_variables)
        summed = [tf.reduce_sum(tf.square(g)) for g in grads]
        return tf.reduce_sum(tf.where(tf.math.is_nan(summed), tf.zeros_like(summed), summed))
    
    @tf.function
    def fim_func_flat(model, x, y):
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            y_pred = model(x, training=False)
            #select a random class uniformly
            selected = tf.random.uniform(shape=(1,), minval=0, maxval=10, dtype=tf.int32)
            output = tf.gather(y_pred, selected, axis=1)
            grads = tape.gradient(tf.math.log(output), model.trainable_variables)
        summed = [tf.reduce_sum(tf.square(g)) for g in grads]
        return tf.reduce_sum(tf.where(tf.math.is_nan(summed), tf.zeros_like(summed), summed))
        
    if FIM_type not in ["stat", "flat", "emp"]:
        raise ValueError("FIM_type must be 'stat', 'flat', or 'emp'")
    total_fim = 0
    for i in range(len(x_all)):
        x= tf.convert_to_tensor(x_all[i], dtype=tf.float32)
        y= tf.convert_to_tensor(y_all[i], dtype=tf.float32)
        x = tf.expand_dims(x, axis=0)  # Add batch dimension
        y = tf.expand_dims(y, axis=0)  # Add batch dimension
        # Compute FIM for each sample
        if FIM_type == "stat":
            #select output based on the output distribution
            total_fim += fim_func_stat(model, x, y)
        elif FIM_type == "flat":
            #select a random class uniformly
            total_fim += fim_func_flat(model, x, y)
        elif FIM_type == "emp":
            #select the label classes
            total_fim += fim_func_emp(model, x, y)
        else:
            raise ValueError("FIM_type must be 'stat', 'flat', or 'emp'")

    fim = total_fim / len(x_all)
    return fim

# Function to compute the average softmax output
def compute_avg_softmax(model, x_all, y_all):
    y_pred = model(x_all, training=False)
    #find the maximum softmax output
    y_pred_max = tf.reduce_mean(tf.reduce_max(y_pred, axis=1))
    #find the average of the maximum softmax output
    y_pred_avg = tf.reduce_mean(y_pred_max)
    #find the minimum softmax output
    y_pred_min = tf.reduce_mean(tf.reduce_min(y_pred, axis=1))
    #find the avg of the label classes
    y_pred_label = tf.reduce_mean(tf.gather(y_pred, tf.argmax(y_all, axis=1), axis=1))
    return {"y_pred_avg": y_pred_avg,
            "y_pred_max": y_pred_max,
            "y_pred_min": y_pred_min,
            "y_pred_label": y_pred_label}

# Training loop
def train_model():
    model = create_model()
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    epochs = 100
    batch_size = 32

    # Metrics to record
    hist = []
    hist.append({"Epoch": 0, "Accuracy": 0, "Test_Accuracy": 0, "stat_FIM": 0, "emp_FIM": 0, "flat_FIM": 0})
    atexit.register(save_metrics_on_exit, hist)
    try:
        for epoch in range(epochs):
            epoch_hist = {}
            print(f"Epoch {epoch + 1}/{epochs}")

            epoch_hist["Epoch"] = epoch + 1

            # Train the model
            history = model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=1)

            # Record train accuracy
            epoch_hist["Accuracy"] = history.history['accuracy'][0]

            # Evaluate on test data
            _, epoch_hist["Test_Accuracy"] = model.evaluate(x_test, y_test, verbose=0)

            # Compute statistical FIM
            epoch_hist["stat_FIM"] = compute_fim(model, x_train[:1000], y_train[:1000], FIM_type="stat")
            
            # Compute empirical FIM
            epoch_hist["emp_FIM"] = compute_fim(model, x_train[:1000], y_train[:1000], FIM_type="emp")

            # COmpute flat FIM
            epoch_hist["flat_FIM"] = compute_fim(model, x_train[:1000], y_train[:1000], FIM_type="flat")

            # Compute average softmax output
            avg_softmax = compute_avg_softmax(model, x_train[:1000],y_train[:1000])
            epoch_hist = {**epoch_hist, **avg_softmax}
            hist.append(epoch_hist)
            print(epoch_hist)
        save_metrics_on_exit(hist)
    except:
        print("Training interrupted. Saving metrics...")
        save_metrics_on_exit(hist)
        raise 
    finally:
        print("Training complete. Saving metrics...")
        save_metrics_on_exit(hist)

def save_metrics_on_exit(metrics):
    df = pd.DataFrame(metrics)
    df.to_csv("metrics.csv", index=False)


def build_graphs():
    metrics = pd.read_csv("metrics.csv")
    #drop first row
    metrics = metrics.drop(0)
    def convert_to_float(x):
        x = x.split("tf.Tensor(")[-1]
        x = x.split(",")[0]
        x = float(x)
        return x
    #cast everything to float
    for col in metrics.columns:
        if col in ["stat_FIM", "emp_FIM", "flat_FIM", "y_pred_avg", "y_pred_max", "y_pred_min", "y_pred_label"]:
            metrics[col] = metrics[col].apply(lambda x: convert_to_float(x))
    print(metrics)
    # Plotting code here
    #test and train accuracy
    plt.figure(figsize=(6, 4))
    plt.plot(metrics["Epoch"], metrics["Accuracy"], label="Train Accuracy")
    plt.plot(metrics["Epoch"], metrics["Test_Accuracy"], label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    #plt.title("Train and Test Accuracy")
    plt.legend()
    plt.savefig("a_accuracy.png")
    plt.close()

    #FIM
    plt.figure(figsize=(6, 4))
    plt.plot(metrics["Epoch"], metrics["stat_FIM"], label="Statistical FIM")
    plt.plot(metrics["Epoch"], metrics["emp_FIM"], label="Empirical FIM")
    plt.plot(metrics["Epoch"], metrics["flat_FIM"], label="Flat FIM")
    plt.xlabel("Epoch")
    plt.ylabel("FIM")
    plt.yscale("log")
    #plt.title("Fisher Information Matrix Approximations")
    plt.legend()
    plt.savefig("a_FIM.png")
    plt.close()
    #Average softmax output
    plt.figure(figsize=(6, 4))
    #plt.plot(metrics["Epoch"], metrics["y_pred_avg"], label="Average Softmax Output")
    plt.plot(metrics["Epoch"], metrics["y_pred_max"], label="Max Softmax Output")
    plt.plot(metrics["Epoch"], metrics["y_pred_min"], label="Min Softmax Output")
    plt.plot(metrics["Epoch"], metrics["y_pred_label"], label="Label Softmax Output")
    plt.xlabel("Epoch")
    plt.ylabel("Softmax Output")
    plt.yscale("log")
    #plt.title("Average Softmax Output")
    plt.legend()
    plt.savefig("a_softmax.png")
    plt.close()

if __name__ == "__main__":
    #train_model()
    build_graphs()
    