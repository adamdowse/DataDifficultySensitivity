
import os
import pandas as pd
import numpy as np
import shutil
import cv2
from sklearn import model_selection
import tensorflow as tf
import tensorflow_datasets as tfds
import PIL


#build and save the img dataset in the correct format
def build_HAM(root_dir,preaugment_size=1000):
    root_dir = os.path.join(data_dir,"HAM10000")
    meta_data_dir = os.path.join(root_dir,"HAM10000_metadata.csv")
    data_dir = os.path.join(root_dir,"data")

    train_size = 0.85 #TODO CHECK THIS
    print("Train size: ",train_size)


    #check if the augmented data exists
    if os.path.exists(os.path.join(root_dir,"C_"+str(preaugment_size))):
        raise ValueError("The augmented data does exist, this does not need to be run")
    #check if the original data exists
    if not os.path.exists(meta_data_dir):
        raise ValueError("The original metadata csv does not exist, please check the path")
    #check if the original data exists
    if not os.path.exists(data_dir):
        raise ValueError("The original data file does not exist, please check the path")
    
    #adapted from SoftAttention paper
    df = pd.read_csv(meta_data_dir)
    df_count = df.groupby('lesion_id').count()
    df_count = df_count[df_count['dx'] == 1]
    df_count.reset_index(inplace=True)

    def duplicates(x):
        unique = set(df_count['lesion_id'])
        if x in unique:
            return 'no' 
        else:
            return 'duplicates'
    
    df['is_duplicate'] = df['lesion_id'].apply(duplicates)
    df_count = df[df['is_duplicate'] == 'no']
    train, test_df = model_selection.train_test_split(df_count, test_size=1-train_size, stratify=df_count['dx'])

    def identify_trainOrtest(x):
        test_data = set(test_df['image_id'])
        if str(x) in test_data:
            return 'test'
        else:
            return 'train'

    #creating df for train and test
    df['train_test_split'] = df['image_id'].apply(identify_trainOrtest)
    train_df = df[df['train_test_split'] == 'train'].copy()
    test_df = df[df['train_test_split'] == 'test'].copy()

    # Image id of train and test images
    train_list = list(train_df['image_id'])
    test_list = list(test_df['image_id'])

    print('Train DF sise: ', len(train_list))
    print(train_df.head())
    print('Test DF size: ', len(test_list))
    print(test_df.head())

    #calculate number of images in each class
    classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    num_classes = len(classes)
    print('Number of classes: ', num_classes)
    train_class_count = {'akiec':0, 'bcc':0, 'bkl':0, 'df':0, 'mel':0, 'nv':0, 'vasc':0}
    train_df.set_index('image_id', inplace=True)
    for img in train_list:
        label = train_df.loc[img, 'dx']
        train_class_count[label] += 1
    print('Original Train Class count: ', train_class_count)

    test_class_count = {'akiec':0, 'bcc':0, 'bkl':0, 'df':0, 'mel':0, 'nv':0, 'vasc':0}
    test_df.set_index('image_id', inplace=True)
    for img in test_list:
        label = test_df.loc[img, 'dx']
        test_class_count[label] += 1
    
    print('Original Test Class count: ', test_class_count)

    # Set the image_id as the index in data_pd
    df.set_index('image_id', inplace=True)

    #if preaugment is true then augment and save images to directory and update df with new files
    if preaugment_size == 0:
        print('No Preaugmentation')
        #copy all training images to a new folder
        if os.path.exists(os.path.join(root_dir,'C_'+str(preaugment_size))):
            shutil.rmtree(os.path.join(root_dir,'C_'+str(preaugment_size)))
        os.mkdir(os.path.join(root_dir,'C_'+str(preaugment_size)))
        for img in train_list:
            shutil.copy(os.path.join(root_dir,'data',img+'.jpg'),os.path.join(root_dir,'C_'+str(preaugment_size),img+'.jpg'))

    else:
        print('Preaugmenting images')
        
        #count remaining images to add
        remaining_class_count = train_class_count.copy()
        for i in classes:
            remaining_class_count[i] = preaugment_size - train_class_count[i]
        print('Remaining Class Count To Add: ', remaining_class_count)

        #make temp folder for aug images
        if os.path.exists(os.path.join(root_dir,'aug_dir')):
            shutil.rmtree(os.path.join(root_dir,'aug_dir'))
        os.mkdir(os.path.join(root_dir,'aug_dir'))
        for i in classes:
            os.mkdir(os.path.join(root_dir,'aug_dir',i))

        #create final data folder
        if os.path.exists(os.path.join(root_dir,'C_'+str(preaugment_size))):
            shutil.rmtree(os.path.join(root_dir,'C_'+str(preaugment_size)))
        os.mkdir(os.path.join(root_dir,'C_'+str(preaugment_size)))

        #Augment imgs
        for class_name in classes:
            print('Aug Class: ',class_name)
            #make temp folder for original images
            if os.path.exists(os.path.join(root_dir,'temp_dir')):
                shutil.rmtree(os.path.join(root_dir,'temp_dir'))
            os.mkdir(os.path.join(root_dir,'temp_dir'))
            os.mkdir(os.path.join(root_dir,'temp_dir',class_name))

            #add original imgs to temp folders with class structure
            class_df = train_df.loc[train_df['dx']==class_name].copy()
            class_list = class_df.index
            for img in class_list:
                file_name = img+'.jpg'
                img_path = os.path.join(root_dir,'data',file_name)
                target_path = os.path.join(root_dir,'temp_dir',class_name,file_name)
                shutil.copy(img_path,target_path)

            #create data generator
            data_aug_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=180,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='nearest'
            )
            #create class seperated data generators
            source_path = os.path.join(root_dir,'temp_dir')
            save_path = os.path.join(root_dir,'aug_dir',class_name)

            #add agumented images to folder to make up a specified number
            num_new_items = remaining_class_count[class_name]
            source_list = os.listdir(os.path.join(source_path,class_name))
            for i in range(0,num_new_items):
                #choose random file to augment
                f_name = source_list[np.random.randint(0,len(source_list))]
                img = cv2.imread(os.path.join(source_path,class_name,f_name))
                aug_img_it = data_aug_gen.flow(x=np.expand_dims(img,0),batch_size=1)
                aug_img = next(aug_img_it)
                cv2.imwrite(os.path.join(save_path,'aug'+str(np.random.randint(100000))+'_'+f_name),np.squeeze(aug_img,axis=0))

            print('-->AugImgs: ',len(os.listdir(save_path)))
            print('--> OGImgs: ',len(os.listdir(os.path.join(source_path,class_name))))


            #transfer all of aug class to thefinal data folder
            for img in os.listdir(save_path):
                shutil.move(os.path.join(save_path,img),os.path.join(root_dir,'C_'+str(preaugment_size),img))

            #transfer all of og class to thefinal data folder
            s_path = os.path.join(source_path,class_name)
            for img in os.listdir(s_path):
                shutil.move(os.path.join(s_path,img),os.path.join(root_dir,'C_'+str(preaugment_size),img))

        #delete temp_dir
        shutil.rmtree(os.path.join(root_dir,'temp_dir'))

        #delete aug_dir
        shutil.rmtree(os.path.join(root_dir,'aug_dir'))

        #We now have a folder called 'C_1000' with 1000 images per class all in one folder
        #update the train_df with the new images
        #get list of images
        train_list = os.listdir(os.path.join(root_dir,'C_'+str(preaugment_size)))
        #if the image is not in the train_df then add it to the train_df
        for img in train_list:
            img = img[:-4]
            if img not in train_df.index:
                #copy row from index - aug
                m_img = img.split('_')
                m_img = m_img[1] + '_'+ m_img[2]
                new_img_row = train_df.loc[m_img].to_numpy(copy=True)
                #add new row to train_df with new index
                train_df.loc[img] = new_img_row
        
    #update train_class_count
    train_class_count = {'akiec':0, 'bcc':0, 'bkl':0, 'df':0, 'mel':0, 'nv':0, 'vasc':0}
    
    train_list = list(train_df.index)
    train_class_count = {'akiec':0, 'bcc':0, 'bkl':0, 'df':0, 'mel':0, 'nv':0, 'vasc':0}
    for img in train_list:
        label = train_df.loc[img, 'dx']
        train_class_count[label] += 1
    print('Updated Train Class count via dir: ', train_class_count)

    print('Total Train Imgs: ',len(train_list))
    data_dir = os.path.join(root_dir,'C_'+str(preaugment_size))

    #change df to [id, img_name, label] format
    train_df.reset_index(inplace=True)
    train_df = train_df[['image_id','dx']]
    train_df.columns = ['image_id','label']
    test_df.reset_index(inplace=True)
    test_df = test_df[['image_id','dx']]
    test_df.columns = ['image_id','label']
    train_df.to_csv(os.path.join(root_dir,'C_'+str(preaugment_size)+'trainmetadata.csv'))
    test_df.to_csv(os.path.join(root_dir,'C_'+str(preaugment_size)+'testmetadata.csv'))

def build_CIFAR10(root_dir,preaugment_size=0):
    #save the cifar10 tfds dataset to the folder and create the metadata csv files
    root_dir = os.path.join(root_dir,"CIFAR10")
    data_dir = os.path.join(root_dir,"C_"+str(preaugment_size))
    test_dir = os.path.join(root_dir,"data")
    
    #check if the augmented data exists
    if os.path.exists(data_dir):
        raise ValueError("The augmented data does exist, this does not need to be run")
    
    #build the data dir
    os.mkdir(data_dir)

    #create the metadata csv file
    train_df = pd.DataFrame(columns=['image_id','label'])

    #download the train dataset
    cifar10_ds, info = tfds.load('cifar10',split=['train'],data_dir=root_dir,with_info=True,download=True,as_supervised=True,shuffle_files=False)
    iterator = iter(cifar10_ds)
    class_names = info.features['label'].names
    c = 0
    for x,y in next(iterator):
        name = str(class_names[y.numpy()])+'_'+str(c)
        PIL.Image.fromarray(x.numpy()).save(os.path.join(data_dir,name +'.jpg'))
        train_df.loc[c] = [name,class_names[y.numpy()]]
        c += 1
    print(train_df.head())

    #create the test metadata csv file
    test_df = pd.DataFrame(columns=['image_id','label'])

    #create the test file dir
    os.mkdir(test_dir)

    #download the test dataset
    cifar10_ds, info = tfds.load('cifar10',split=['test'],data_dir=root_dir,with_info=True,download=True,as_supervised=True,shuffle_files=False)
    iterator = iter(cifar10_ds)
    class_names = info.features['label'].names
    c = 0
    for x,y in next(iterator):
        name = str(class_names[y.numpy()])+'_'+str(c)
        PIL.Image.fromarray(x.numpy()).save(os.path.join(test_dir,name +'.jpg'))
        test_df.loc[c] = [name,class_names[y.numpy()]]
        c += 1
    print(test_df.head())

    #save the metadata csv files
    train_df.to_csv(os.path.join(root_dir,"C_"+str(preaugment_size)+"trainmetadata.csv"))
    test_df.to_csv(os.path.join(root_dir,"C_"+str(preaugment_size)+"testmetadata.csv"))


if __name__ == "__main__":
    build_CIFAR10('/com.docker.devenvironments.code/',preaugment_size=0)
