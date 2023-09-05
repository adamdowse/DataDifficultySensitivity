#This is a custom version of the ImageDataGenerator class from Keras. It is used to generate batches of tensor image data with real-time data augmentation. The data will be looped over (in batches).
#It will be able to select images from a directory and mask images to select based on a metric that is updated every epoch. 



import numpy as np
import keras
import tensorflow as tf
import os
import pandas as pd
from sklearn import model_selection
import shutil
import cv2
import time

class CustomImageGen(tf.keras.utils.Sequence):
    def __init__(self, DS_name, root_dir, meta_data_dir, batch_size, img_size,acc_sample_weight, mask_metric='loss',trainsize=0.85,model_name='IRv2',force=False):
        #get directory of images (no label seperation just a folder of images)
        #get csv file of labels
        #TODO condence to use config

        self.DS_name = DS_name #name of dataset
        self.model_name = model_name #name of model to use for preprocessing
        self.root_dir = root_dir #directory with folder of images inside Root_dir/DS_name/data/imgs.jpg
        self.meta_data_dir = meta_data_dir #directory of meta data with labels info in csv format
        self.batch_size = batch_size #batch size
        self.img_size = img_size #image size
        self.mask_metric = mask_metric #metric to select images based on
        self.trainsize = trainsize #percentage of images to use for training
        self.epoch_num = 0 #current epoch number
        self.adjusted_epoch_num = 0 #current epoch number adjusted for method changes (can be float)
        self.batch_num = 0 #current batch number
        self.acc_sample_weight = acc_sample_weight
        
        #ensure all images are in the csv file and return the csv file as a pandas dataframe
        #train_df is [index, image_id, label]
        self.train_df,self.test_df,self.data_dir, self.test_data_dir = self.preprocess_csv(self.DS_name,
                                                                                           self.meta_data_dir,
                                                                                           self.root_dir,
                                                                                           preaugment=True,
                                                                                           preaugment_size=1000,
                                                                                           force=force)
        
        self.indexes = self.train_df.index.values #indexes of images in the train set

    def update_losses(self,model):
        #return the losses for all images in the train set
        self.losses = np.zeros(len(self.indexes))
        self.update_mask(method='All')
        for i in range(self.num_train_batches):
            imgs,labels = self.__getitem__(i,training_set=True)
            self.losses[i*self.batch_size:(i+1)*self.batch_size] = model.get_items_loss(imgs,labels,training=False)
        

    def update_mask(self,method='All',split_type=None,percentage=None,stage=None):
        #update the mask array based on the new metric
        #get the new metric
        #update the mask array
        print('Updating Mask with method: ',method,' and split type: ',split_type,' and percentage: ',percentage,' and stage: ',stage)
        t = time.time()
        if method == 'All':
            #keep all images in the mask
            self.mask = np.ones(len(self.indexes))

        elif method == 'Loss':
            #seperate the mask based on the loss
            if split_type == 'High':
                #keep the top percentage of images
                self.mask = np.zeros(len(self.indexes))
                #get the indexes of the top percentage of images
                top_loss_indexes = np.argsort(self.losses)[-int(percentage*len(self.indexes)):]
                #set the mask to 1 for the top percentage of images
                self.mask[top_loss_indexes] = 1
            elif split_type == 'Low':
                #keep the bottom percentage of images
                self.mask = np.zeros(len(self.indexes))
                #get the indexes of the bottom percentage of images
                bottom_loss_indexes = np.argsort(self.losses)[:int(percentage*len(self.indexes))]
                #set the mask to 1 for the bottom percentage of images
                self.mask[bottom_loss_indexes] = 1
            elif split_type == 'Staged':
                #keep the images with loss between stages
                self.mask = np.zeros(len(self.indexes))
                #get the indexes of the images with loss between stages
                stage_loss_indexes = np.argsort(self.losses)[stage*int(percentage*len(self.indexes)):(stage+1)*int(percentage*len(self.indexes))]
                #set the mask to 1 for the images with loss between stages
                self.mask[stage_loss_indexes] = 1
        else:
            print('Invalid Mask Update Method')
        print('Mask Update Time: ',time.time()-t)
        
    def build_batches(self,batch_size=None):
        print('Building Batches')
        #this is run before the start of each traingin cycle, inclusing test set
        #update the mask array based on the new metric
        #create the indexes array from df that contain the image ids and labels
        #TODO add option to just build train or test batches
        if batch_size is None:
            batch_size = self.batch_size

        self.indexes = self.train_df.index.values
        self.indexes = self.indexes[self.mask.astype(bool)]
        num_images = len(self.indexes)
            
        #shuffle and batch the indexes
        np.random.shuffle(self.indexes)
        if num_images % batch_size == 0:
            self.num_train_batches = num_images // batch_size
        else:
            self.num_train_batches = (num_images // batch_size) + 1
        split_points = [i*batch_size for i in range(1,self.num_train_batches)]
        self.indexes = np.array_split(self.indexes,split_points)
        print('--> Num Batches: ',self.num_train_batches, 'Num Images: ',num_images, 'Batch Size: ',batch_size)

        #test batches
        self.test_indexes = self.test_df.index.values
        np.random.shuffle(self.test_indexes)
        num_images = len(self.test_indexes)
        if num_images % batch_size == 0:
            self.num_test_batches = num_images // batch_size
        else:
            self.num_test_batches = (num_images // batch_size) + 1
        split_points = [i*batch_size for i in range(1,self.num_test_batches)]
        self.test_indexes = np.array_split(self.test_indexes,split_points)


            

    def on_epoch_end(self,method='Vanilla'):
        #update the epoch number
        self.epoch_num += 1
        if method == 'HighLossPercentage':
            self.epoch_num_adjusted += 1 #TODO CHANGE THIS
            self.batch_num += self.num_train_batches
        elif method == 'Vanilla':
            self.epoch_num_adjusted += 1
            self.batch_num += self.num_train_batches

    def __getitem__(self, index, training_set=True):
        #return the batch as a tensor of images and labels
        if training_set:
            df = self.train_df
            di = self.data_dir
            ind = self.indexes
        else:
            df = self.test_df
            di = self.test_data_dir
            ind = self.test_indexes
        #get the indexes for the batch
        batch_indexes = ind[index]

        #load the images
        batch_images = []
        for i in batch_indexes:
            img = tf.keras.preprocessing.image.load_img(os.path.join(di,i+'.jpg'),target_size=self.img_size)
            img = tf.keras.preprocessing.image.img_to_array(img)
            batch_images.append(img)
        batch_images = tf.convert_to_tensor(batch_images)

        #get the labels from the indexes
        batch_labels = df.loc[batch_indexes,'label'].to_numpy()
        #convert to indecies
        for i in range(len(batch_labels)):
            batch_labels[i] = self.class_names.index(batch_labels[i])
        batch_labels = tf.convert_to_tensor(batch_labels,dtype=tf.int32)

        #preprocess step
        if self.model_name == 'IRv2':
            #preprocess for inception resnet v2 (this does it 'sample-wise' TODO CHECK THIS)
            batch_images = tf.keras.applications.inception_resnet_v2.preprocess_input(batch_images)
            batch_labels = tf.one_hot(batch_labels,self.num_classes)
        
        return batch_images, batch_labels



    def __len__(self):
        #return the number of batches per epoch
        return self.num_batches


    def preprocess_csv(self, DS_name, meta_data_dir,root_dir,preaugment=True,preaugment_size=1000,force=True):
        #preaugment_size is per class
        #preproces options for individual datasets (add more as needed)
        #convert dataframe into [id, img_name, label] format
        #only do this id the files need to be generated

        np.random.seed(42)


        if DS_name == 'HAM10000':
            #only create new data if force is true or folder or meta does not exist
            if force or not os.path.exists(os.path.join(root_dir,'C_'+str(preaugment_size))):
                #TODO add checks for metadata
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
                train, test_df = model_selection.train_test_split(df_count, test_size=1-self.trainsize, stratify=df_count['dx'])

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
                self.classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
                self.num_classes = len(self.classes)
                self.class_names = self.classes
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
                if preaugment:
                    print('Preaugmenting images')
                    
                    #count remaining images to add
                    remaining_class_count = train_class_count.copy()
                    for i in self.classes:
                        remaining_class_count[i] = preaugment_size - train_class_count[i]
                    print('Remaining Class Count To Add: ', remaining_class_count)

                    #make temp folder for aug images
                    if os.path.exists(os.path.join(root_dir,'aug_dir')):
                        shutil.rmtree(os.path.join(root_dir,'aug_dir'))
                    os.mkdir(os.path.join(root_dir,'aug_dir'))
                    for i in self.classes:
                        os.mkdir(os.path.join(root_dir,'aug_dir',i))

                    #create final data folder
                    if os.path.exists(os.path.join(root_dir,'C_'+str(preaugment_size))):
                        shutil.rmtree(os.path.join(root_dir,'C_'+str(preaugment_size)))
                    os.mkdir(os.path.join(root_dir,'C_'+str(preaugment_size)))

                    #Augment imgs
                    for class_name in self.classes:
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
                        #TODO add option to reduce large classes to wanted image count
                        #aug_datagen = data_aug_gen.flow_from_directory(source_path,save_to_dir=save_path,save_format='jpg',save_prefix='aug',target_size=top_img_size,batch_size=bs,shuffle=True)
                        
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
                    
                    for img in train_list:
                        label = train_df.loc[img[:-4], 'dx']
                        train_class_count[label] += 1
                    print('Updated Train Class count via dir: ', train_class_count)

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

                test_data_dir = os.path.join(root_dir,'data')
                #Calculate any normalization values here
                #TODO

                #img_size=(299,299)
            
            else:
                print('Data path exist')
                train_df = pd.read_csv(os.path.join(root_dir,'C_'+str(preaugment_size)+'trainmetadata.csv'))
                test_df = pd.read_csv(os.path.join(root_dir,'C_'+str(preaugment_size)+'testmetadata.csv'))
                data_dir = os.path.join(root_dir,'C_'+str(preaugment_size))
                test_data_dir = os.path.join(root_dir,'data')
                train_df.set_index('image_id', inplace=True)
                test_df.set_index('image_id', inplace=True)

                train_class_count = {'akiec':0, 'bcc':0, 'bkl':0, 'df':0, 'mel':0, 'nv':0, 'vasc':0}
                train_list = os.listdir(data_dir)
                print('Total Train Img: ',len(train_list))
                for img in train_list:
                    label = train_df.loc[img[:-4], 'label']
                    train_class_count[label] += 1
                print('Train Class count via dir: ', train_class_count)

                train_class_count = {'akiec':0, 'bcc':0, 'bkl':0, 'df':0, 'mel':0, 'nv':0, 'vasc':0}
                train_list = list(train_df.index)
                for img in train_list:
                    label = train_df.loc[img, 'label']
                    train_class_count[label] += 1
                print('Train Class count via DataFrame: ', train_class_count)

                test_class_count = {'akiec':0, 'bcc':0, 'bkl':0, 'df':0, 'mel':0, 'nv':0, 'vasc':0}
                test_list = list(test_df.index)
                for img in test_list:
                    label = test_df.loc[img, 'label']
                    test_class_count[label] += 1
                print('Test Class count: ', test_class_count)

                self.num_classes = 7
                self.class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
            
            self.acc_sample_weight = [[self.acc_sample_weight[i]] for i in range(len(self.acc_sample_weight))]
            self.acc_sample_weight = np.array([self.acc_sample_weight for i in range(self.batch_size)])
            print('--> Weighted Train Acc Sample Weight shape: ',self.acc_sample_weight.shape)

            return train_df, test_df, data_dir, test_data_dir



                

            #ADD WAY OF DOING DATA AUG PRE TRAINING OR TO DO IT IN THE GENERATOR (would need to update train_list and test_list)



