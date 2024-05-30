#running kmeans and calculating FIM



import tensorflow as tf
import numpy as np
import os
import wandb
import time
import pandas as pd
import matplotlib.pyplot as plt


ds = pd.read_csv('WineClusteringDS.csv')
col_names = ds.columns[4:]

ds = ds.drop(col_names,axis=1)

#normalise data to 0-1
ds = (ds-ds.min())/(ds.max()-ds.min())
print(ds.head())


#print graph of data
# plt.scatter(ds['Alcohol'],ds['Malic_Acid'])
# plt.xlabel('Alcohol')
# plt.ylabel('Malic acid')


num_cols = len(ds.columns)



k = 3
#Kmeans
def KmeansInit(ds,k,num_cols):
    #initilise clusters
    clusters = {}
    
    for i in range(k):
        center = np.random.rand(num_cols)
        points = []
        cluster = {'center':center,'points':points}
        clusters[i] = cluster
    return clusters


def distance(x,y):
    return np.linalg.norm(x-y)

def assign_clusters(X, clusters):
    for idx in range(X.shape[0]):
        dist = []
        
        curr_x = X[idx]
        
        for i in range(k):
            dis = distance(curr_x,clusters[i]['center'])
            dist.append(dis)
        curr_cluster = np.argmin(dist)
        clusters[curr_cluster]['points'].append(curr_x)
    return clusters

def update_clusters(X, clusters):
    for i in range(k):
        points = np.array(clusters[i]['points'])
        if points.shape[0] > 0:
            new_center = points.mean(axis =0)
            clusters[i]['center'] = new_center
            
            clusters[i]['points'] = []
    return clusters

def update_clusters_lr(X, clusters,lr):
    for i in range(k):
        points = np.array(clusters[i]['points'])
        if points.shape[0] > 0:
            new_center = clusters[i]['center'] + (points.mean(axis =0)-clusters[i]['center'])*lr
            clusters[i]['center'] = new_center
            
            clusters[i]['points'] = []
    return clusters

def pred_cluster(X, clusters):
    pred = []
    for i in range(X.shape[0]):
        dist = []
        for j in range(k):
            dist.append(distance(X[i],clusters[j]['center']))
        pred.append(np.argmin(dist))
    return pred

def calc_loss(X, clusters):
    loss = 0
    for i in range(k):
        points = np.array(clusters[i]['points'])
        if points.shape[0] > 0:
            loss_k = np.zeros(points.shape[0])
            for j in range(points.shape[0]):
                loss_k[j] = distance(points[j],clusters[i]['center'])
            loss += np.min(loss_k)
    return loss

def calc_loss_surface(X):
    #for 2 ks
    res = 0.1
    
    loss_surface = np.zeros((int(1/res),int(1/res),int(1/res),int(1/res)))
    #[k1x,k1y,k2x,k2y]
    for i in range(int(1/res)):
        print(i)
        for j in range(int(1/res)):
            for l in range(int(1/res)):
                for m in range(int(1/res)):
                    #calc total loss if clusters are at these points
                    loss = 0
                    c = 0
                    for x in X:
                        loss_k = np.zeros(2)
                        loss_k[0] = distance(x,[i*res,j*res])
                        loss_k[1] = distance(x,[l*res,m*res])
                        loss += np.min(loss_k)
                        c += 1
                    loss_surface[i,j,l,m] = loss/c

    return loss_surface

def calc_FIM(X, clusters):
    g = np.zeros((k,num_cols)) # [k x num_cols]
    counts = np.zeros(k) # [1 x k]
    for x in X: # each data point
        dists = np.zeros(k)
        for i in range(k):
            dists[i] = distance(x,clusters[i]['center'])# dist to each cluster
        dists = dists/np.sum(dists) #to probs
        #print("dists",dists)
        selected = np.random.choice(k,p=dists)#p=dists#selected cluster
        #selected = np.argmin(dists)
        y_hat = clusters[selected]['center'] # [num_cols]
        #print("x",x)
        #print("selected",selected,clusters[selected]['center'])
        counts[selected] += 1

        for j in range(num_cols):
            term1 = -y_hat[j]/distance(x,y_hat)**2
            term2 = (-x[j]-y_hat[j])/distance(x,y_hat)
            g[selected,j] += term1+term2

    FIM = g**2
    FIM = np.sum(FIM,axis=None)

        # g_top = np.sqrt(num_cols)*(x-clusters[selected]['center']) # [num_cols]
        # print(g_top)
        # g_bottom = np.abs(x-clusters[selected]['center']) # [num_cols]
        # print(g_bottom)
        # g = g_top/g_bottom
        #ln_g = -1/(x-y_hat) # [num_cols]
        #FIM += np.sum(np.square(ln_g))

    print("counts",counts)
    print("FIM",FIM)
    FIM = FIM/np.sum(counts)
    print("FIM",FIM)
    return FIM

f,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(10,10))
max_runs = 5
for run in range(max_runs):
    a = 0.3
    clusters = KmeansInit(ds.values,k,num_cols)
    colors = ['r','g','b','y','c','brown','purple','orange','pink','grey']

    for i in range(k):
        if run == max_runs-1:
            ax1.scatter(clusters[i]['center'][0],clusters[i]['center'][1],s=50,c=colors[i],alpha=a,zorder=1,marker='o')
    clusters = assign_clusters(ds.values,clusters)
    losses = []
    FIMs = []
    r =20
    for j in range(r):
        clusters = assign_clusters(ds.values,clusters)
        loss = calc_loss(ds.values,clusters)
        print('Loss: ',loss)
        losses.append(loss)

        FIMs.append(calc_FIM(ds.values,clusters))

        clusters = update_clusters_lr(ds.values,clusters,0.7)
        #clusters = update_clusters(ds.values,clusters)
        if run == max_runs-1:
            a += (1-a)/r
            for i in range(k):
                ax1.scatter(clusters[i]['center'][0],clusters[i]['center'][1],s=50,c=colors[i],alpha=a,zorder=1,marker='o')

    ax2.plot(losses)
    ax3.plot(FIMs)

pred = pred_cluster(ds.values,clusters)
pred = [colors[i] for i in pred]
ax1.scatter(ds['Alcohol'],ds['Malic_Acid'],c=pred,zorder=0,alpha=0.5,marker='x')

f.savefig('WineClusteringDSOG.png')

#loss surface
#loss_surface = calc_loss_surface(ds.values)
#(ax11,ax12,ax13,ax14,ax21,ax22,ax23,ax24,ax31,ax32,ax33,ax34,ax41,ax42,ax43,ax44)
#f,ax = plt.subplots(3,3,figsize=(20,20))

#s = [3,5,7]
#for i in range(len(s)):
#    for j in range(len(s)):
#       ax[i,j].imshow(loss_surface[:,:,s[i],s[j]])
#f.savefig('WineClusteringDSLossSurface.png')