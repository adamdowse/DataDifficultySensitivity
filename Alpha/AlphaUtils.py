import tensorflow as tf
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

@tf.function
def get_fim_flat_approx_sample_single(self,item,model):
    #FIM flat approximation for a single sample
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
def get_fim_emp_approx_sample_single(self,item,model):
    x,y = item
    x = tf.expand_dims(x,0)
    y = tf.expand_dims(y,0)
    bs = tf.shape(x)[0]
    loss_function = tf.keras.losses.CategoricalCrossentropy()
    
    with tf.GradientTape() as tape:
        #print(text_batch[i])
        y_hat = model(x,training=False) #get model output (softmax) [1 x num_classes]
        loss = loss_function(y,y_hat)
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
def get_fim_stat_approx_sample_single(self,item,model):

    x,y = item
    x = tf.expand_dims(x,0)
    y = tf.expand_dims(y,0)
    bs = tf.shape(x)[0]
    loss_function = tf.keras.losses.CategoricalCrossentropy()
    
    with tf.GradientTape() as tape:
        y_hat = model(x,training=False) #get model output (softmax) [1 x num_classes]
        loss = loss_function(y,y_hat)
        selected = tf.squeeze(tf.random.categorical(tf.math.log(y_hat), bs),axis=1) #sample from the output [BS x 1]
        y_hat = tf.gather(y_hat,selected,axis=1,batch_dims=1) #get the output for the selected class [BS x 1]
        output = tf.math.log(y_hat)#tf.math.log(output) #log the output [BS x 1]

    j = tape.jacobian(output,model.trainable_variables)
    layer_sizes = [tf.reduce_sum(tf.size(v)) for v in model.trainable_variables] #get the size of each layer
    j = [tf.reshape(j[i],(bs,layer_sizes[i])) for i in range(len(j))] #reshape the gradient to [BS x num_layer_params x layers]
    j = tf.concat(j,axis=1) #concat the gradient over the layers [BS x num_params]
    j = tf.square(j) #square the gradient [BS x num_params]
    j = tf.reduce_sum(j,axis=1) #sum the gradient [ 1]
    return j, y_hat,output

def get_fim_dist(dataset,model,approx_type="flat",batch_size=32,limit=None):
    """
    Get the Fisher Information Matrix (FIM) for a dataset and model.
    
    Parameters:
    - dataset: tf.data.Dataset object
    - model: tf.keras.Model object
    - approx_type: str, type of approximation to use ("flat", "empirical", "statistical")
    - batch_size: int, batch size to use for the dataset
    
    Returns:
    - fim: numpy array, Fisher Information Matrix
    """

    
    fim = []
    y_hats = []
    outputs = []
    if approx_type == "flat":
        get_fim_sample = get_fim_flat_approx_sample_single
    elif approx_type == "empirical":
        get_fim_sample = get_fim_emp_approx_sample_single
    elif approx_type == "statistical":
        get_fim_sample = get_fim_stat_approx_sample_single
    else:
        raise ValueError("Unknown approximation type: {}".format(approx_type))
    
    if batch_size is not None:
        raise ValueError("Batch size must be None for FIM calculation. FIM is calculated over the entire dataset.")
    else:
        dataset = dataset.unbatch()
    if limit is None:
        limit = len(dataset)
    
    classes = []
    colors = []
    for i, item in enumerate(tqdm(dataset,total=limit, desc="Calculating FIM")):
        if i >= limit:
            break
        j, y_hat, output = get_fim_sample(None, item, model)
        fim.append(j.numpy())
        y_hats.append(y_hat.numpy())
        outputs.append(output.numpy())
        classes.append(np.argmax(item[1].numpy()))  # Collect the class index for each sample
        colors.append(item[0].numpy()[0][0][0])  # Collect the color for each sample (if needed)

    # Concatenate the FIM for all samples
    fim = np.concatenate(fim, axis=0)
    y_hats = np.concatenate(y_hats, axis=0)
    outputs = np.concatenate(outputs, axis=0)
    classes = np.array(classes)
    
    return fim, y_hats, outputs,classes,colors

def calc_alpha(dataset, model, alpha_type="flat", batch_size=32,limit=None,fig=None,ds_name=None):
    """
    Calculate the alpha value for a dataset and model.
    
    Parameters:
    - dataset: tf.data.Dataset object
    - model: tf.keras.Model object
    - alpha_type: str, type of alpha calculation to use ("flat", "empirical", "statistical")
    - batch_size: int, batch size to use for the dataset
    
    Returns:
    - alpha: float, alpha value
    """
    
    fims, y_hats, outputs,classes = get_fim_dist(dataset,
                                        model,
                                        approx_type=alpha_type,
                                        batch_size=batch_size,
                                        limit=limit)

    if fig is not None:
        plt.scatter(outputs, fims, label=f"Alpha {ds_name}")
    
    # Calculate the alpha value as the mean of the FIM
    def _alpha_func(x,a):
        return a*np.log(1+x)**2
    popt, pcov = curve_fit(_alpha_func, y_hats,fims)
    alpha = popt[0]
    return alpha

def calc_alpha_percentiles(dataset, model, alpha_type="flat", batch_size=32, limit=None,ax=None, ds_name=None,og_alpha=None):
    # ...existing code to collect x_vals and y_vals...
    def _alpha_func(x,a):
        return a*np.log(1+x)**2
    fims, y_hats, outputs ,classes,colours= get_fim_dist(dataset,
                                        model,
                                        approx_type=alpha_type,
                                        batch_size=batch_size,
                                        limit=limit)
    outputs = -outputs

    # if ax is not None:
    #     ax[0].scatter(outputs, fims, label=f"Alpha {ds_name}")


    # Bin the data if needed (for example, using np.digitize or np.histogram)
    bins = np.linspace(outputs.min(), outputs.max(), num=20)
    digitized = np.digitize(outputs, bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    y_5th = []
    y_95th = []
    y_mean = []

    for i in range(1, len(bins)):
        ys = fims[digitized == i]
        if len(ys) > 0:
            y_5th.append(np.percentile(ys, 5))
            y_95th.append(np.percentile(ys, 95))
            y_mean.append(np.mean(ys))
        else:
            y_5th.append(np.nan)
            y_95th.append(np.nan)
            y_mean.append(np.nan)

    # Remove bins with nan values
    valid = ~np.isnan(y_mean)
    bin_centers = bin_centers[valid]
    y_mean = np.array(y_mean)[valid]
    y_5th = np.array(y_5th)[valid]
    y_95th = np.array(y_95th)[valid]

    # Fit your curve (replace `your_curve_func` with your actual fitting function)
    popt_mean, _ = curve_fit(_alpha_func, bin_centers, y_mean)
    popt_5th, _ = curve_fit(_alpha_func, bin_centers, y_5th)
    popt_95th, _ = curve_fit(_alpha_func, bin_centers, y_95th)

    #draw the 3 largest difference to the alpha line
    if ax is not None:
        if og_alpha is not None:
            diffs = fims - _alpha_func(outputs, og_alpha[0])
        else:
            diffs = fims - _alpha_func(outputs, popt_mean[0])
        #normalize the diffs by the standard deviation at that y_hat value
        norm_factor = _alpha_func(outputs, popt_95th[0]) - _alpha_func(outputs, popt_5th[0])
        diffs = diffs / norm_factor
        if ds_name == "filled" and False:
            #plot each class in a different color
            unique_cols = np.unique(classes)
            print(f"Unique classes in {ds_name}: {unique_cols}")
            for col in unique_cols:
                mask = classes == col
                ax.scatter(outputs[mask], fims[mask], label=f"Alpha {ds_name} Class {col}", alpha=0.8)
        elif ds_name == "filled" and True:
            #plot the points based on the color of the data
            cmap = plt.get_cmap('viridis')
            norm = plt.Normalize(vmin=0, vmax=1)
            cols = np.array(colours)
            scatter = ax.scatter(outputs, fims, c=cols, cmap=cmap, norm=norm, label=f"Alpha {ds_name}", alpha=0.1)
            # Add a colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Color Value')




        else:
            ax.scatter(outputs, fims, label=f"Alpha {ds_name}",alpha=0.1)

        #get the indices of the largest and smallest diffs
        n = 2
        largest_indices = np.argsort(np.abs(diffs))[-n:]
        smallest_indices = np.argsort(np.abs(diffs))[:n]
        max_imgs =[]
        for idx in largest_indices:
            max_imgs.append((dataset.unbatch().skip(idx).take(1).get_single_element()[0], diffs[idx]))
        min_imgs = []
        for idx in smallest_indices:
            min_imgs.append((dataset.unbatch().skip(idx).take(1).get_single_element()[0], diffs[idx]))
    else:
        max_imgs = []
        min_imgs = []
            
    return (popt_5th[0],popt_mean[0],  popt_95th[0]), np.mean(y_hats), np.mean(outputs),(max_imgs, min_imgs)

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

class GSLRscheduleCBV2(tf.keras.callbacks.Callback):
    def __init__(self, k ):
        super(GSLRscheduleCBV2, self).__init__()
        #decay the learning rate by k if the average gradient scale decreased last epoch
        #this looks within the epoch to calc grad scale change
        self.k = k

    def on_epoch_end(self, epoch, logs=None):
        #update at the end of each epoch
        first_half = self.model.optimizer.grad_scale_1 / (self.model.optimizer.total_batches//2)
        second_half = self.model.optimizer.grad_scale_2 / (self.model.optimizer.total_batches//2)
        tf.print("First half: ",first_half," Second half: ",second_half)
        if first_half > second_half:
            #if the gradient scale decreased, decay the learning rate
            self.model.optimizer.learning_rate = self.model.optimizer.learning_rate * self.k
            #reset the grad scales
        self.model.optimizer.grad_scale_1.assign(0.0)
        self.model.optimizer.grad_scale_2.assign(0.0)
        self.model.optimizer.batch_count.assign(0)

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

# class GSLRschedule(tf.keras.optimizers.SGD):
#     def __init__(self, initial_learning_rate=0.01):
#         super(GSLRschedule, self).__init__()
#         self.learning_rate = initial_learning_rate
#         self.grad_scale_1 = tf.Variable(0.0)
#         self.grad_scale_2 = tf.Variable(0.0)
#         self.total_batches = 1641
#         self.batch_count = tf.Variable(0)


#     def __call__(self, step):
#         return self.learning_rate


class WandBEOELog(tf.keras.callbacks.Callback):
    def __init__(self):
        """
        Callback to log the end of epoch metrics to wandb
        """
        super(CustomCallback, self).__init__()
        pass
    
    def on_epoch_end(self, epoch, logs=None):
        #save metrics to wandb
        #get most recent logs
        print(logs)
        wandb.log(logs,commit=True)

    


def plot_alpha_imgs(info={}):
    keys = list(info.keys())

    fig, img_ax = plt.subplots(len(keys), 5, figsize=(10, 10))
    for i in range(2):
        for j in range(len(keys)):
            img_ax[j, i].imshow(info[keys[j]][0][i][0])
            img_ax[j, i].axis('off')
            img_ax[j, i].set_title(f"{keys[j]} Img \n {info[keys[j]][0][i][1]:.5f}")

        # img_ax[0, i].imshow(info["train"][0][i][0])
        # img_ax[0, i].axis('off')
        # img_ax[0, i].set_title(f"Train Img \n {info['train'][0][i][1]:.5f}")

        # img_ax[1, i].imshow(info["inv_train_imgs"][0][i][0])
        # img_ax[1, i].axis('off')
        # img_ax[1, i].set_title(f"Inv Train Img \n {info['inv_train_imgs'][0][i][1]:.5f}")

        # img_ax[2, i].imshow(info["test_imgs"][0][i][0])
        # img_ax[2, i].axis('off')
        # img_ax[2, i].set_title(f"Test Img \n {info['test_imgs'][0][i][1]:.5f}")

        # img_ax[3, i].imshow(info["inv_test_imgs"][0][i][0])
        # img_ax[3, i].axis('off')
        # img_ax[3, i].set_title(f"Inv Test Img \n {info['inv_test_imgs'][0][i][1]:.5f}")
    for i in range(3,5):
        for j in range(len(keys)):
            img_ax[j, i].imshow(info[keys[j]][1][i-3][0])
            img_ax[j, i].axis('off')
            img_ax[j, i].set_title(f"{keys[j]} Img \n {info[keys[j]][1][i-3][1]:.5f}")

        # img_ax[1, i].imshow(info["inv_train_imgs"][1][i-3][0])
        # img_ax[1, i].axis('off')
        # img_ax[1, i].set_title(f"Inv Train Img \n {info['inv_train_imgs'][1][i-3][1]:.5f}")

        # img_ax[2, i].imshow(info["test_imgs"][1][i-3][0])
        # img_ax[2, i].axis('off')
        # img_ax[2, i].set_title(f"Test Img \n {info['test_imgs'][1][i-3][1]:.5f}")

        # img_ax[3, i].imshow(info["inv_test_imgs"][1][i-3][0])
        # img_ax[3, i].axis('off')
        # img_ax[3, i].set_title(f"Inv Test Img \n {info['inv_test_imgs'][1][i-3][1]:.5f}")
    plt.tight_layout()
    plt.savefig("alpha_imgs_comparison.png")






