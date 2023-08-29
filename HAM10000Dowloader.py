import numpy as np
import pandas as pd
import os
import scipy as sp
import shutil
import tensorflow as tf
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile


# Path: HAM10000Dowloader.py


def download_data():
    data_urls = ['https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000/download?datasetVersionNumber=2']
    extract_path = '/tmp/HAM10000'
    for url in data_urls:
        with urlopen(url) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall(extract_path)

def process_data():
    #REMEMBER TO CHANGE THE PATH
    train_dir = "data/train"
    test_dir = "data/test"
    aug_dir = 'data/aug_dir' #this directory is temporary and will be deleted after the augmented images are generated
    
    data_pd = pd.read_csv("HAM10000_metadata.csv")

    df_count = data_pd.groupby('lesion_id').count()
    df_count = df_count[df_count['dx'] == 1]
    df_count.reset_index(inplace=True)

    def duplicates(x):
        unique = set(df_count['lesion_id'])
        if x in unique:
            return 'no' 
        else:
            return 'duplicates'
    
    data_pd['is_duplicate'] = data_pd['lesion_id'].apply(duplicates)
    data_pd.head()

    df_count = data_pd[data_pd['is_duplicate'] == 'no']
    train, test_df = sp.model_selection.train_test_split(df_count, test_size=0.15, stratify=df_count['dx'])

    def identify_trainOrtest(x):
        test_data = set(test_df['image_id'])
        if str(x) in test_data:
            return 'test'
        else:
            return 'train'

    #creating train_df
    data_pd['train_test_split'] = data_pd['image_id'].apply(identify_trainOrtest)
    train_df = data_pd[data_pd['train_test_split'] == 'train']

    # Image id of train and test images
    train_list = list(train_df['image_id'])
    test_list = list(test_df['image_id'])

    print('Train DF sise: ', len(train_list))
    train_df.head()
    print('Test DF size: ', len(test_list))
    test_df.head()

    # Set the image_id as the index in data_pd
    data_pd.set_index('image_id', inplace=True)

    #Build file structure
    os.mkdir(train_dir)
    os.mkdir(test_dir)
    targetnames = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    for i in targetnames:
        directory1=train_dir+'/'+i
        directory2=test_dir+'/'+i
        os.mkdir(directory1)
        os.mkdir(directory2)

    for image in train_list:
        file_name = image+'.jpg'
        label = data_pd.loc[image, 'dx']

        # path of source image 
        source = os.path.join('HAM10000', file_name)

        # copying the image from the source to target file
        target = os.path.join(train_dir, label, file_name)

        shutil.copyfile(source, target)

    for image in test_list:

        file_name = image+'.jpg'
        label = data_pd.loc[image, 'dx']

        # path of source image 
        source = os.path.join('HAM10000', file_name)

        # copying the image from the source to target file
        target = os.path.join(test_dir, label, file_name)

        shutil.copyfile(source, target)

    # Augmenting images and storing them in temporary directories 
    for img_class in targetnames:

        #creating temporary directories
        # creating a base directory
        os.mkdir(aug_dir)
        # creating a subdirectory inside the base directory for images of the same class
        img_dir = os.path.join(aug_dir, 'img_dir')
        os.mkdir(img_dir)

        img_list = os.listdir('HAM10000/train_dir/' + img_class)

        # Copy images from the class train dir to the img_dir 
        for file_name in img_list:
            # path of source image in training directory
            source = os.path.join('HAM10000/train_dir/' + img_class, file_name)

            # creating a target directory to send images 
            target = os.path.join(img_dir, file_name)

            # copying the image from the source to target file
            shutil.copyfile(source, target)

        # Temporary augumented dataset directory.
        source_path = aug_dir

        # Augmented images will be saved to training directory
        save_path = 'HAM10000/train_dir/' + img_class

        # Creating Image Data Generator to augment images
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(

            rotation_range=180,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'

        )

        batch_size = 50

        aug_datagen = datagen.flow_from_directory(source_path,save_to_dir=save_path,save_format='jpg',target_size=(299, 299),batch_size=batch_size)

        # Generate the augmented images
        aug_images = 8000 

        num_files = len(os.listdir(img_dir))
        num_batches = int(np.ceil((aug_images - num_files) / batch_size))

        # creating 8000 augmented images per class
        for i in range(0, num_batches):
            images, labels = next(aug_datagen)

        # delete temporary directory 
        shutil.rmtree('aug_dir')
    
    
if __name__ == "__main__":
    download_data()
    #process_data()







