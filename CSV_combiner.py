import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt



#take the csv files in the directory and combine them into a single dataframe

#get each csv as a dataframe

csv_root = 'CombinedData/CombinedCSV/CombinedCSV'
csv_files = os.listdir(csv_root)
df = pd.read_csv(os.path.join(csv_root,csv_files[0]),index_col=0)
df = df.drop(columns=['thresh_diff_01_0','thresh_diff_001_0'])

#make the values of final_losses_0 NaN if split0 is not train
df_train = df.copy()
df_test = df.copy()
df_train['final_losses_0'] = df_train['final_losses_0'].where(df_train['split0'] == 'train',other=None)
df_test['final_losses_0'] = df_test['final_losses_0'].where(df_test['split0'] == 'test',other=None)

df_train = df_train.drop(columns=['split0'])
df_test = df_test.drop(columns=['split0'])

print(df.head())
print(df_train.head())
print(df_test.head())

for i in range(1,len(csv_files)):
    df2 = pd.read_csv(os.path.join(csv_root,csv_files[i]),index_col=0)
    df2 = df2.drop(columns=['path','label','thresh_diff_01_0','thresh_diff_001_0'])
    df2_train = df2.copy()
    df2_test = df2.copy()
    df2_train['final_losses_0'] = df2_train['final_losses_0'].where(df2_train['split0'] == 'train',other=None)
    df2_test['final_losses_0'] = df2_test['final_losses_0'].where(df2_test['split0'] == 'test',other=None)
    df2_train = df2_train.drop(columns=['split0'])
    df2_test = df2_test.drop(columns=['split0'])
    df2_train = df2_train.rename(columns={'final_losses_0':'final_losses_'+str(i)})
    df2_test = df2_test.rename(columns={'final_losses_0':'final_losses_'+str(i)})
    print(df2_train.head())
    print(df2_test.head())
    #print('Nans')
    #print(df2_train['final_losses_'+str(i)].isna().sum())
    #print(df2_test['final_losses_'+str(i)].isna().sum())
    # print('Counts')
    # print(df2_train['final_losses_'+str(i)].count()+df2_test['final_losses_'+str(i)].count())
    # print(df2_test['final_losses_'+str(i)].count())
    print('Zeros')
    #count the zeros
    print((df2_train['final_losses_'+str(i)] > 0).sum())
    df_train = pd.concat([df_train,df2_train],axis=1)
    df_test = pd.concat([df_test,df2_test],axis=1)

#average the final losses
df_train['final_losses'] = df_train[['final_losses_'+str(i) for i in range(len(csv_files))]].mean(axis=1,skipna=True)
df_test['final_losses'] = df_test[['final_losses_'+str(i) for i in range(len(csv_files))]].mean(axis=1,skipna=True)

#standard deviation of the final losses
df_train['final_losses_std'] = df_train[['final_losses_'+str(i) for i in range(len(csv_files))]].std(axis=1,skipna=True)
df_test['final_losses_std'] = df_test[['final_losses_'+str(i) for i in range(len(csv_files))]].std(axis=1,skipna=True)

print(df.head())
print(df_train.head())
print(df_test.head())

#save the combined dataframes
df_train.to_csv('CombinedData/CombinedTrainData.csv')
df_test.to_csv('CombinedData/CombinedTestData.csv')

#count nans in the final losses
print('Train NaNs:',df_train['final_losses'].isna().sum())
print('Test NaNs:',df_test['final_losses'].isna().sum())

print('Train zeros:',(df_train['final_losses'] == 0).sum())
print('Test zeros:',(df_test['final_losses'] == 0).sum())

print('Train count:',df_train['final_losses'].count())
print('Test count:',df_test['final_losses'].count())

#make the nans in the final losses 0
df_train['final_losses'] = df_train['final_losses'].fillna(0)
df_test['final_losses'] = df_test['final_losses'].fillna(0)

#remove the zeros from both dataframes
gzero_index = (df_train['final_losses'] > 0) & (df_test['final_losses'] > 0)
df_test = df_test[gzero_index]
df_train = df_train[gzero_index]

#plot a 2d histogram of the final losses

plt.hist2d(np.log(df_train['final_losses']),np.log(df_test['final_losses']),bins=50,norm=plt.cm.colors.LogNorm())
plt.xlabel('Train Final Losses')
#get the current ticks
xticks = [1e-8,1e-6,1e-4,1e-2,1,100]
#convert the ticks to the original values
plt.xticks(np.log(xticks),xticks)
plt.yticks(np.log(xticks),xticks)
plt.ylabel('Test Final Losses')
plt.title('Train vs Test Final Losses')
plt.colorbar()
plt.savefig('CombinedData/TrainVsTestFinalLossesHist.png')
plt.clf()

#plot scatter of each class
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.hist2d(np.log(df_train['final_losses'][df_train['label'] == i]),np.log(df_test['final_losses'][df_test['label'] == i]),bins=50,norm=plt.cm.colors.LogNorm())
    plt.plot([np.log(1e-8),np.log(100)],[np.log(1e-8),np.log(100)],color='red',linestyle='--',alpha=0.5)
    plt.xlabel(class_names[i])
    plt.xlim(np.log(1e-8),np.log(100))
    plt.ylim(np.log(1e-8),np.log(100))
    plt.xticks([])
    plt.yticks([])

plt.savefig('CombinedData/TrainVsTestFinalLossesByClass.png')
plt.clf()



#find the data points closesnt to the corners and center
locs = [(df_train['final_losses'].min(), df_test['final_losses'].min()), 
           (df_train['final_losses'].min(), df_test['final_losses'].max()), 
           (df_train['final_losses'].max(), df_test['final_losses'].min()), 
           (df_train['final_losses'].max(), df_test['final_losses'].max()),
           (np.exp(np.log(df_train['final_losses']).min()+(np.log(df_train['final_losses']).max()-np.log(df_train['final_losses']).min())/2),
           np.exp(np.log(df_test['final_losses']).min()+(np.log(df_test['final_losses']).max()-np.log(df_test['final_losses']).min())/2)),
           (np.exp(np.log(df_train['final_losses']).min()+(np.log(df_train['final_losses']).max()-np.log(df_train['final_losses']).min())*0.25),
           np.exp(np.log(df_test['final_losses']).min()+(np.log(df_test['final_losses']).max()-np.log(df_test['final_losses']).min())*0.25)),
           (np.exp(np.log(df_train['final_losses']).min()+(np.log(df_train['final_losses']).max()-np.log(df_train['final_losses']).min())*0.75),
           np.exp(np.log(df_test['final_losses']).min()+(np.log(df_test['final_losses']).max()-np.log(df_test['final_losses']).min())*0.75))]

# n = 5
# top_df = []
# for loc in locs:
#     #find the closest n points to the location
#     dist = np.sqrt((df_train['final_losses']-loc[0])**2 + (df_test['final_losses']-loc[1])**2)
#     idx = np.argpartition(dist,n)
#     print('Closest to:',loc)
#     temp_df = df_train[['path','label','final_losses']].iloc[idx[:n]]
#     temp_df['test_final_losses'] = df_test['final_losses'].iloc[idx[:n]]
#     print(temp_df)
#     top_df.append(temp_df)

n = 5
top_df = []
for loc in locs:
    #find the closest n points to the location
    dist = np.sqrt((np.log(df_train['final_losses'])-np.log(loc[0]))**2 + (np.log(df_test['final_losses'])-np.log(loc[1]))**2)
    idx = np.argpartition(dist,n)
    print('Closest to:',loc)
    temp_df = df_train[['path','label','final_losses']].iloc[idx[:n]]
    temp_df['test_final_losses'] = df_test['final_losses'].iloc[idx[:n]]
    print(temp_df)
    top_df.append(temp_df)

#plot the images of the closest points
plt.scatter(np.log(df_train['final_losses']),np.log(df_test['final_losses']),label='All Data')
for i in range(len(locs)):
    plt.scatter(np.log(top_df[i]['final_losses']),np.log(top_df[i]['test_final_losses']))
plt.xlabel('Train Final Losses')
plt.ylabel('Test Final Losses')
plt.title('Train vs Test Final Losses')
plt.savefig('CombinedData/TrainVsTestFinalLossesClosest.png')
plt.clf()
    




plt.clf()
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
#get the images of the closest points
for i in range(len(locs)):
    for j in range(n):
        plt.subplot(1,n,j+1)
        img_path = top_df[i].path.iloc[j]
        img_path = img_path.split('Combined/')[-1]
        img_path = 'CombinedData/Combined/'+img_path
        plt.imshow(plt.imread(img_path))
        #get the final loss from the train and test dataframes
        plt.title(f'Train:{top_df[i].final_losses.iloc[j]:.3e}\nTest: {top_df[i].test_final_losses.iloc[j]:.3e}',fontsize=8)
        plt.xlabel(class_names[top_df[i].label.iloc[j]])
        #turn off axis ticks
        plt.xticks([])
        plt.yticks([])
    plt.savefig('CombinedData/ClosestPointsImages'+str(i)+'.png')


#scatter plot the final losses train on the y axis and test on the x axis
# plt.scatter(df_train['final_losses'],df_test['final_losses'])
# plt.xlabel('Train Final Losses')
# plt.ylabel('Test Final Losses')
# plt.title('Train vs Test Final Losses')
# #add the standard deviation of the final losses
# plt.errorbar(df_train['final_losses'],df_test['final_losses'],xerr=df_train['final_losses_std'],yerr=df_test['final_losses_std'],fmt='o')
# plt.savefig('CombinedData/TrainVsTestFinalLosses.png')

# plt.clf()
# plt.scatter(df_train['final_losses'],df_train['final_losses_std'])
# plt.xlabel('Train Final Losses')
# plt.ylabel('Train Final Losses STD')
# plt.title('Train Final Losses vs STD')
# plt.savefig('CombinedData/TrainFinalLossesVsSTD.png')

plt.clf()
#scatter plot for one datapoint

#make a dataset for each of splits in the train-test graph



def get_data_in_split(train_df,test_df,closestPoint,percentageSplit,isLog=False):
    #combine the train and test dataframes
    #change the name of the final losses column
    train_df = train_df.rename(columns={'final_losses':'final_losses_train'})
    test_df = test_df.rename(columns={'final_losses':'final_losses_test'})
    df = pd.concat([train_df[['path','label','final_losses_train']],test_df['final_losses_test']],axis=1)
    df['split'] = 'none'
    #add dist column
    df['dist'] = np.sqrt((df['final_losses_train']-closestPoint[0])**2 + (df['final_losses_test']-closestPoint[1])**2)
    df['log_dist'] = np.sqrt((np.log(df['final_losses_train'])-np.log(closestPoint[0]))**2 + (np.log(df['final_losses_test'])-np.log(closestPoint[1]))**2)
    print(df.head())

    #Make sure there is an even split of all classes in each split
    #add a new column
    num_classes = 10
    for c in range(num_classes):
        threshidx = np.argpartition(df['dist'][df['label']==c],int(percentageSplit*df['dist'][df['label']==c].count()))[int(percentageSplit*df['dist'][df['label']==c].count())]
        threshval = df['dist'][df['label']==c].iloc[threshidx]
        print(threshval)
        df['split'][(df['label']==c) & (df['dist'] < threshval)] = 'train'
        df['split'][(df['label']==c) & (df['dist'] >= threshval)] = 'test'
    return df

locs = [(df_train['final_losses'].min(), df_test['final_losses'].min()), 
           (df_train['final_losses'].min(), df_test['final_losses'].max()), 
           (df_train['final_losses'].max(), df_test['final_losses'].min()), 
           (df_train['final_losses'].max(), df_test['final_losses'].max())]

trainPercent = 0.8
#LowAmbiguityTrain = low train low test (normal scaling)
df = get_data_in_split(df_train,df_test,locs[0],trainPercent,isLog=False)
df['path'] = df['path'].apply(lambda x: x.split('Combined/')[-1])
print(df.head())
df.to_csv('CombinedData/LowAmbiguity.csv')

#HighAMbiguityTrain = high train high test (normal scaling)
df = get_data_in_split(df_train,df_test,locs[3],trainPercent,isLog=False)
df['path'] = df['path'].apply(lambda x: x.split('Combined/')[-1])
print(df.head())
df.to_csv('CombinedData/HighAmbiguity.csv')

#OODTrain = low train high test (normal scaling)
df = get_data_in_split(df_train,df_test,locs[1],trainPercent,isLog=False)
df['path'] = df['path'].apply(lambda x: x.split('Combined/')[-1])
df.to_csv('CombinedData/OOD.csv')
print(df.head())
#ComplexTrain = high train low test (normal scaling)
df = get_data_in_split(df_train,df_test,locs[2],trainPercent,isLog=False)
df['path'] = df['path'].apply(lambda x: x.split('Combined/')[-1])
df.to_csv('CombinedData/Complex.csv')
print(df.head())









