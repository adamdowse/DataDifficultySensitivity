#Create 4 datasets based on CIFAR10 classes that better analyse the capabilities of the models
#1. Get CIFAR10 and IMgaenet classes that are the same as the CIFAR10 classes
#2. Combine the datasets into one
#3. Augment the dataset variably so that some data is more difficult to classify
#4. Save the datasets
#5. Take random train test splits of the data and train a collection of models on them and record performance
#6. Score the models with the data difficulty metric for both train and test sets and record the results
#7. visualise the results on train diff vs test diff plots
#8. Create 4 datasets based on plot (each datapoint now has a train and test difficulty score)
#9. Train model on combinations of datasets and record performance


import numpy as np
import os
import tensorflow as tf
import pandas as pd
from PIL import Image
import glob


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 1.
def build_total_dataset(data_root):
    # put all the data into one dataset
    root =data_root # upper data folder
    #downlaod the CINIC10 dataset
    if not os.path.exists(os.path.join(root,'CINIC-10.tar.gz')):
        print('Downloading CINIC10 dataset')
        os.system('wget -c https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz -P ' + root)
        #extract the CINIC10 dataset
        os.system('tar -xvf ' + os.path.join(root, 'CINIC-10.tar.gz') + ' -C ' + root)
        print('CINIC10 dataset downloaded and extracted')
    else:
        print('CINIC10 dataset already exists')
    combined = os.path.join(root, 'Combined')
    if not os.path.exists(combined):
        os.makedirs(combined)
    
    #train images
    #get list of directories in CINIC10
    cinic_test = os.path.join(root, 'test')
    cinic_train = os.path.join(root, 'train')
    cinic_valid = os.path.join(root, 'valid')

    #copy test, train and val folders into combined
    for i in range(10):
        if not os.path.exists(os.path.join(combined, class_names[i])):
            os.makedirs(os.path.join(combined, class_names[i]))

        os.system('cp -r ' + os.path.join(cinic_test, class_names[i]) + '/. ' + os.path.join(combined, class_names[i]))
        os.system('cp -r ' + os.path.join(cinic_train, class_names[i]) + '/. ' + os.path.join(combined, class_names[i]))
        os.system('cp -r ' + os.path.join(cinic_valid, class_names[i]) + '/. ' + os.path.join(combined, class_names[i]))
    print('Combined dataset created')

def convert_imgs():
    #convert the png to have the correct sRGB profile
    root ='CombinedData'
    combined = os.path.join(root, 'Combined')
    #convert all the images to sRGB
    for i in range(10):
        for file in os.listdir(os.path.join(combined, class_names[i])):
            with Image.open(os.path.join(combined, class_names[i], file)) as img:
                #img = img.convert('RGB')
                img.save(os.path.join(combined, class_names[i], file))
        print('Converted: ', class_names[i])

def count_classes():
    #count the number of images in each class
    root ='CombinedData'
    combined = os.path.join(root, 'Combined')

    for i in range(10):
        print('Combined: ', len(os.listdir(os.path.join(combined, class_names[i]))))

def create_random_split_dataset():
    #create random splits of the data in train and test and record the data in a csv file
    combined_dir = 'CombinedData/Combined'

    #create a df with all the image paths, label and the split it is in
    df = pd.DataFrame(columns=['path', 'label', 'split0'])

    path_list = []
    label_list = []

    for i in range(10):
        for file in os.listdir(os.path.join(combined_dir, class_names[i])):
            path_list.append(os.path.join(combined_dir, class_names[i], file))
            label_list.append(i)

    split_list = np.random.choice(['train', 'test'], len(path_list), p=[0.5, 0.5])

    df['path'] = path_list
    df['label'] = label_list
    df['split0'] = split_list

    train_path_list = np.array(df['path'][df['split0'] == 'train'])
    test_path_list = np.array(df['path'][df['split0'] == 'test'])
    train_label_list = np.array(df['label'][df['split0'] == 'train'])
    test_label_list = np.array(df['label'][df['split0'] == 'test'])

    ds = tf.data.Dataset.from_tensor_slices((path_list, label_list))
    train_ds = tf.data.Dataset.from_tensor_slices((train_path_list, train_label_list))
    test_ds = tf.data.Dataset.from_tensor_slices((test_path_list, test_label_list))

    def process_path(file_path, label):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, [32, 32])
        img = tf.image.convert_image_dtype(img, tf.float32)
        img /= 255.0
        return img, label, file_path

    def redu(img, label, file_path):
        return img, label

    train_ds = train_ds.map(process_path) #now we have a dataset of images, labels and file paths
    test_ds = test_ds.map(process_path)
    ds = ds.map(process_path)

    train_ds = train_ds.map(redu)
    test_ds = test_ds.map(redu)

    return train_ds, test_ds,ds,df,path_list,label_list

def create_model(model_type):
    if model_type == 'rand':
        moodel_type = np.random.choice(['ResNet', 'CNN'])
    if model_type == 'ResNet':
        model = tf.keras.applications.ResNet50(include_top=False, weights=None, classes=10, input_shape=(32, 32, 3))
    elif model_type == 'CNN':
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10)
        ])
    elif model_type == 'VGG':
        model = tf.keras.applications.VGG16(include_top=True, weights=None, classes=10)
    
    opt = tf.keras.optimizers.SGD(learning_rate=np.random.uniform(0.001, 0.1), momentum=np.random.uniform(0.0, 0.9))
    model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model


def main(data_root, output_root,build_data):
    if build_data:
        build_total_dataset(data_root)
        convert_imgs(data_root)
    count_classes()
    train_ds,test_ds,ds,df,path_list,label_list = create_random_split_dataset()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    #create model
    model = create_model(model_type='CNN') #creates a desired model with random hyperparameters in range
    model.summary()

    #define data difficulty metric
    class diff_callback(tf.keras.callbacks.Callback):
        def __init__(self, ds):
            self.ds = ds # these are ordered
            self.thresh_diff_01 = np.zeros(270000)
            self.thresh_diff_001 = np.zeros(270000)

            self.red_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)

            self.c01 = 0
            self.c001 = 0

        def on_epoch_end(self, epoch, logs=None):
            #threshold difficulty metric
            i=0
            for batch in self.ds:
                img, label, path = batch
                with tf.GradientTape() as tape:
                    pred = self.model(img, training=False)
                    loss = self.red_loss(label, pred)
                for j in range(len(loss)):
                    if loss[j] < 0.01 and self.thresh_diff_01[i] == 0:
                        self.thresh_diff_01[i] = epoch
                        self.c01 += 1
                    if loss[j] < 0.001 and self.thresh_diff_001[i] == 0:
                        self.thresh_diff_001[i] = epoch
                        self.c001 += 1
                    i+=1
            
            print('Threshold difficulty metric counts: ', self.c01, self.c001)

    class CustomEarlyStopping(tf.keras.callbacks.Callback):
        def __init__(self, ds, monitor='val_loss', patience=5, min_delta=0):
            super(CustomEarlyStopping, self).__init__()
            self.monitor = monitor
            self.patience = patience
            self.min_delta = min_delta
            self.wait = 0
            self.best = np.Inf
            self.stopped_epoch = 0
            self.ds = ds
            self.final_losses = np.zeros(270000)

        def on_epoch_end(self, epoch, logs=None):
            current = logs.get(self.monitor)
            if current is None:
                return

            if np.less(current - self.min_delta, self.best):
                self.best = current
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True

        def on_train_end(self, logs=None):
            if self.stopped_epoch > 0:
                print(f'Early stopping at epoch {self.stopped_epoch + 1}')

            i = 0
            for img, label, path in self.ds:
                loss,metric =self.model.evaluate(img, label)
                self.final_losses[i] = loss
                i+=1

    ds = ds.batch(32)

    diff_cb = diff_callback(ds)
    early_cb = CustomEarlyStopping(ds)

    

    train_ds = train_ds.shuffle(buffer_size=270000).batch(32)
    test_ds = test_ds.shuffle(buffer_size=270000).batch(32)

    model.fit(train_ds, epochs=250, callbacks=[diff_cb, early_cb], validation_data=test_ds,shuffle=True)
            

    #update df
    df['thresh_diff_01_'+str(0)] = diff_cb.thresh_diff_01
    df['thresh_diff_001_'+str(0)] = diff_cb.thresh_diff_001
    df['final_losses_'+str(0)] = early_cb.final_losses

    #save the df
    run_id = str(np.random.uniform(0,100000))
    df.to_csv(os.path.join(output_root,'diff_metrics_'+run_id+'.csv'))
    #save text file with hyperparameters and final loss and accuracy
    

    #TODO need to do this many times and combine the data. It would be best if we could do this in parallel
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Root dirs')
    parser.add_argument('-data', type=str, default='', help='root of the Combined CINIC10 dataset')
    parser.add_argument('-output', type=str, default='', help='root of the folder to save the data')
    parser.add_argument('-build', type=bool, default=False, help='whether to build the data')
    args = parser.parse_args()
    main(args.data, args.output, args.build)
    
