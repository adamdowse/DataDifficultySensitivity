import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def loss_fim_gap(file_name):
    #find the maximum gap in the FIM wtr the loss
    df = pd.read_csv(file_name)

    fim_idx = [1,3,5,7,9]
    loss_idx = [2,4,6,8,10]
    epochs = [0,20,40,60,80]

    run_id = 4

    losses = df[str(loss_idx[run_id])]
    fims = df[str(fim_idx[run_id])]

    losses = np.log10(losses)
    fims = np.log10(fims)

    max_loss = np.max(losses)
    min_loss = np.min(losses)
    loss_bins = np.linspace(min_loss,max_loss,50)
    maxfim_gap = np.zeros(len(loss_bins)-1)
    for i in range(len(maxfim_gap)):
        maxfim_gap[i] = np.max(fims[(losses>loss_bins[i]) & (losses<loss_bins[i+1])]) - np.min(fims[(losses>loss_bins[i]) & (losses<loss_bins[i+1])])
    
    max_fim = np.max(fims)
    min_fim = np.min(fims)
    fim_bins = np.linspace(min_fim,max_fim,50)
    maxloss_gap = np.zeros(len(fim_bins)-1)
    for i in range(len(maxloss_gap)):
        maxloss_gap[i] = np.max(losses[(fims>fim_bins[i]) & (fims<fim_bins[i+1])]) - np.min(losses[(fims>fim_bins[i]) & (fims<fim_bins[i+1])])
    

    plt.plot(loss_bins[:-1],maxfim_gap,label="Max FIM Gap")
    plt.plot(maxloss_gap,fim_bins[:-1],label="Max Loss Gap")
    #plt.xscale("log")
    #plt.yscale("log")
    plt.xlabel("Loss")
    plt.ylabel("Max FIM Gap")
    plt.title("Max FIM Gap vs Loss")
    plt.grid()
    plt.savefig("MaxFIMGap.png")
    plt.clf()


def csv2_to_graphs(file_name,output_name):
    df = pd.read_csv(file_name)

    fim_idx = [1,4,7,10,13]
    loss_idx = [2,5,8,11,14]
    norm_idx = [3,6,9,12,15]
    epochs = [0,20,40,60,80]
    for i in range(len(fim_idx)):
        fig, ((ax1,ax2,ax3)) = plt.subplots(1,3,figsize=(30,10))
        ax1.scatter(df[str(fim_idx[i])],df[str(loss_idx[i])],s=1,alpha=0.2,color="red")
        ax1.set_xlabel("FIM")
        ax1.set_ylabel("Loss")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlim(10e-10,10e5)
        ax1.set_ylim(10e-8,10e1)
        ax1.grid()

        ax2.scatter(df[str(fim_idx[i])],df[str(norm_idx[i])],s=1,alpha=0.2,color="blue")
        ax2.set_xlabel("FIM")
        ax2.set_ylabel("Grad Norm")
        ax2.set_xscale("log")
        ax2.set_yscale("log") 
        ax2.set_xlim(10e-10,10e5)
        ax2.set_ylim(10e-8,10e3)
        ax2.grid()

        ax3.scatter(df[str(loss_idx[i])],df[str(norm_idx[i])],s=1,alpha=0.2,color="green")
        ax3.set_xlabel("Loss")
        ax3.set_ylabel("Grad Norm")
        ax3.set_xscale("log")
        ax3.set_yscale("log") 
        ax3.set_xlim(10e-10,10e5)
        ax3.set_ylim(10e-8,10e3)
        ax3.grid()

        plt.savefig(str(output_name)+str(epochs[i])+".png")
        #clear plot
        plt.clf()


def csv_to_graphs(file_name,output_name):
    df = pd.read_csv(file_name)
    #df2 = pd.read_csv("RandomCorrectLossFIM.csv")
    fim_idx = [1,3,5,7,9]
    loss_idx = [2,4,6,8,10]
    epochs = [0,20,40,60,80]
    for i in range(len(fim_idx)):
        plt.scatter(df[str(fim_idx[i])],df[str(loss_idx[i])],s=1,alpha=0.2,color="red")
        #plt.scatter(df2[str(fim_idx[i])],df2[str(loss_idx[i])],s=1,alpha=0.2,color="green")
        #plt.xlabel("FIM")
        plt.xlabel("FIM")
        plt.ylabel("Loss")
        #plt.title("FIM vs Loss at epoch "+str(epochs[i]))
        plt.title("FIM vs Loss at epoch "+str(epochs[i]))
        #log x axis
        plt.xscale("log")
        plt.yscale("log")
        #axis limits
        plt.xlim(10e-10,10e5) #(10e-10,10e5)
        plt.ylim(10e-8,10e1) #(10e-8,10e1)
        
        #add grid lines
        plt.grid()
        plt.savefig(str(output_name)+str(epochs[i])+".png")
        #clear plot
        plt.clf()

def batch_csv(file_name,output_name):
    df = pd.read_csv(file_name)
    #df2 = pd.read_csv("RandomCorrectLossFIM.csv")
    
    losses = df.iloc[:,1::2]
    losses = losses.drop(columns=["0"])
    losses = losses.iloc[:,:100]
    fims = df.iloc[:,0::2]
    fims = fims.drop(columns=["Unnamed: 0"])
    fims = fims.iloc[:,:100]
    #only take the first row
    losses = losses.iloc[0]
    fims = fims.iloc[0]
    print(losses.head())
    print(fims.head())
    cmap = plt.get_cmap('PiYG')
    norm = plt.Normalize(vmin=0, vmax=len(fims) - 1)
    #for i in range(len(fims) - 1):
    #    plt.plot(fims[i:i+2], losses[i:i+2], color=cmap(norm(i)), alpha=0.25)

    fig, ax = plt.subplots()
    
    # Plot each segment with an arrow
    for i in range(len(fims) - 1):
        arrow = FancyArrowPatch((fims[i], losses[i]), (fims[i+1], losses[i+1]),
                                color=cmap(norm(i)), alpha=1, arrowstyle='-|>', mutation_scale=10)
        ax.add_patch(arrow)
    
    plt.xlabel("FIM")
    plt.ylabel("Loss")
    plt.title("FIM vs Loss")
    #log x axis
    plt.xscale("log")
    plt.yscale("log")

    #plt.legend()

    #add grid lines
    plt.grid()
    plt.savefig(str(output_name)+".png")
    #clear plot
    plt.clf()

    for i in range(len(fims)):
        plt.scatter(fims[i], losses[i], color=cmap(norm(i)), alpha=0.25)
    plt.xlabel("FIM")
    plt.ylabel("Loss")
    plt.title("FIM vs Loss")
    #log x axis
    plt.xscale("log")
    plt.yscale("log")
    #axis limits

    #add grid lines
    plt.grid()
    plt.savefig(str(output_name)+"Scatter.png")

def alpha_csv(file_name,output_name):
    df = pd.read_csv(file_name)


    fim_idx = [1,4,7,10,13]
    loss_idx = [2,5,8,11,14]
    r_idx = [3,6,9,12,15]
    epochs = [0,20,40,60,80]

    def col(x):
        #x is between 0 and 1
        # return an rgb color from red to blue (red =1) (blue = 0)
        return (x,0,1-x)

    for i in range(len(fim_idx)):
        for j in range(4000):
            plt.scatter(df[str(fim_idx[i])][j],df[str(loss_idx[i])][j],s=1,alpha=0.5,color=col(df[str(r_idx[i])][j]))
        #plt.scatter(df2[str(fim_idx[i])],df2[str(loss_idx[i])],s=1,alpha=0.2,color="green")
        #plt.xlabel("FIM")
        plt.xlabel("FIM")
        plt.ylabel("Loss")
        #plt.title("FIM vs Loss at epoch "+str(epochs[i]))
        plt.title("FIM vs Loss at epoch "+str(epochs[i]))
        #log x axis
        plt.xscale("log")
        plt.yscale("log")
        #axis limits
        plt.xlim(10e-10,10e5) #(10e-10,10e5)
        plt.ylim(10e-8,10e1) #(10e-8,10e1)
        
        #add grid lines
        plt.grid()
        plt.savefig(str(output_name)+str(epochs[i])+".png")
        #clear plot
        plt.clf()

def corrGraphs(file_name,output_name):
    df = pd.read_csv(file_name)
    #df2 = pd.read_csv("RandomCorrectLossFIM.csv")
    fim_idx = [1,4,7,10,13]
    loss_idx = [2,5,8,11,14]
    corr_idx = [3,6,9,12,15]
    epochs = [0,20,40,60,80]
    for i in range(len(fim_idx)):
        #get the losses and fim values when corr is 1
        corr1_losses = df[str(loss_idx[i])][df[str(corr_idx[i])]==1]

        plt.scatter(df[str(fim_idx[i])][df[str(corr_idx[i])]==1],df[str(loss_idx[i])][df[str(corr_idx[i])]==1],s=1,alpha=0.2,color="green")
        plt.scatter(df[str(fim_idx[i])][df[str(corr_idx[i])]==0],df[str(loss_idx[i])][df[str(corr_idx[i])]==0],s=1,alpha=0.2,color="red")
        plt.xlabel("FIM")
        plt.ylabel("Loss")
        plt.title("FIM vs Loss at epoch "+str(epochs[i]))
        #log x axis
        plt.xscale("log")
        plt.yscale("log")
        #axis limits
        plt.xlim(10e-10,10e5) #(10e-10,10e5)
        plt.ylim(10e-8,10e1) #(10e-8,10e1)
        
        #add grid lines
        plt.grid()
        plt.savefig(str(output_name)+str(epochs[i])+".png")
        #clear plot
        plt.clf()

if __name__ == "__main__":
    corrGraphs("NormalSelectedTestLossFIM.csv","NormalSelectedTestLossFIM")
    #batch_csv("NormalBatch80LossFIM.csv","NormalBatch80LossFIM")
    #loss_fim_gap("DenseNormalLossFIM.csv")
    #csv2_to_graphs("MagNormalLossFIM.csv","MagNormal")
    #alpha_csv("AugmentTradNoiseLossFIM.csv","AugmentTradNoiseLossFIM")

    #csv_to_graphs("AugmentTradFlipyLossFIM.csv","AugmentTradFlipyLossFIM")