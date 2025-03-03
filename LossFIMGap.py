import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.optimize import curve_fit


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
    fim_idx = [1,3,5,7,9,11]
    loss_idx = [2,4,6,8,10,12]
    epochs = [0,20,40,60,80,100]
    for i in range(len(fim_idx)):
        plt.scatter(df[str(fim_idx[i])],(df[str(loss_idx[i])]),s=1,alpha=0.2,color="red")
        #plt.scatter(df2[str(fim_idx[i])],df2[str(loss_idx[i])],s=1,alpha=0.2,color="green")
        #plt.xlabel("FIM")
        plt.xlabel("FIM")
        plt.ylabel("-log(y_hat)")
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

        #plt.scatter(df[str(fim_idx[i])][df[str(corr_idx[i])]==1],df[str(loss_idx[i])][df[str(corr_idx[i])]==1],s=1,alpha=0.2,color="green")
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

def DisjointGraphs(file_name,output_name):
    df = pd.read_csv(file_name)
    #df2 = pd.read_csv("RandomCorrectLossFIM.csv")
    fim_idx =    [1,5,9,13,17]
    fimsel_idx = [2,6,10,14,18]
    y_idx =      [3,7,11,15,19]
    ysel_idx =  [4,8,12,16,20]

    epochs = [0,20,40,60]
    for i in range(len(epochs)):
        #get the losses and fim values when corr is 1
        #corr1_losses = df[str(loss_idx[i])][df[str(corr_idx[i])]==1]

        #calc alpha 
        def func(x,a):
            return a*np.log(1+x)**2
        def func2(x,a):
            return a*np.log(1-np.log(1-np.exp(-x)))**2

        #get the y values and FIM when the FIM is calculated with the same y_hat (FIM L=1, y_hat L=1) and (FIM L=0, y_hat L=0)
        df_epoch = df[[str(fim_idx[i]), str(y_idx[i])]][(df[str(fimsel_idx[i])]==0) & (df[str(ysel_idx[i])]==0)]
        df_epoch = pd.concat([df_epoch, df[[str(fim_idx[i]), str(y_idx[i])]][(df[str(fimsel_idx[i])]==0) & (df[str(ysel_idx[i])]==0)]])
        df_epoch = df_epoch.dropna()
        print(df_epoch.head())
        popt, pcov = curve_fit(func, -df_epoch[str(y_idx[i])].values, df_epoch[str(fim_idx[i])].values)
        print(popt)


        fimsel=[1,0,1,0]
        ysel=[1,1,0,0]
        colors=["green","blue","orange","red"]
        for j in range(len(fimsel)):
            plt.scatter(-df[str(y_idx[i])][(df[str(fimsel_idx[i])]==fimsel[j])&(df[str(ysel_idx[i])]==ysel[j])],df[str(fim_idx[i])][(df[str(fimsel_idx[i])]==fimsel[j])&(df[str(ysel_idx[i])]==ysel[j])],s=1,alpha=0.2,color=colors[j],label="FIM L="+str(fimsel[j])+", y_hat L="+str(ysel[j]))
        
        x = np.logspace(-7,1.5,1000)
        plt.plot(x,func(x,*popt),label="FIM=alog(1-log(y_hat))^2")
        plt.plot(x,func2(x,*popt),label="FIM=alog(1-log(1-y_hat))^2\na="+str(popt[0]))

        plt.xlabel("-log(y_hat)")
        plt.ylabel("FIM")
        plt.title("FIM vs -log(y_hat) at epoch "+str(epochs[i]))
        #log x axis
        plt.xscale("log")
        plt.yscale("log")
        #axis limits
        plt.xlim(10e-8,10e2) #(10e-10,10e5)
        plt.ylim(10e-8,10e5) #(10e-8,10e1)
        #plot legend middle left
        plt.legend(loc="center left")

        #add grid lines
        plt.grid()
        plt.savefig(str(output_name)+str(epochs[i])+".png")
        #clear plot
        plt.clf()

def YhatYhat(file_name,output_name):
    df = pd.read_csv(file_name)
    #df2 = pd.read_csv("RandomCorrectLossFIM.csv")
    y1_idx =    [1,4,7,10,13]
    y2_idx =    [2,5,8,11,14]
    sel_idx =   [3,6,9,12,15]

    epochs = [0,20,40,60,80]
    for i in range(len(y1_idx)):
        #get the losses and fim values when corr is 1
        #corr1_losses = df[str(loss_idx[i])][df[str(corr_idx[i])]==1]
        colors=["green","blue","orange","red"]
        reduced_df =df[[str(y1_idx[i]),str(y2_idx[i]),str(sel_idx[i])]]
        #remove the rows that select the same class
        current_df = reduced_df[[str(y1_idx[i]),str(y2_idx[i])]][reduced_df[str(sel_idx[i])]==0] #-log(y_hat)
        same_df =    reduced_df[[str(y1_idx[i]),str(y2_idx[i])]][reduced_df[str(sel_idx[i])]==1]
        #remove rows that have values less than 0.1 or greater than 0.9
        lowlim=-np.log(0.999)
        uplim=-np.log(0.001)
        current_df = current_df[(current_df[str(y1_idx[i])]>lowlim) &(current_df[str(y1_idx[i])]<uplim)&(current_df[str(y2_idx[i])]>lowlim)&(current_df[str(y2_idx[i])]<uplim)]

        x = np.logspace(-4,2,100)
        y = -np.log(1- np.exp(-x))
        plt.plot(x,y,label="y=-log(1-e^(-x))",alpha=0.5)
        plt.plot(x,x,label="y=x",alpha=0.5,color="black")

        #plt.scatter(np.exp(-df[str(y1_idx[i])][(df[str(sel_idx[i])]==1)]),np.exp(-df[str(y2_idx[i])][(df[str(sel_idx[i])]==1)]),s=1,alpha=0.2,color="red")
        #plt.scatter(df[str(y1_idx[i])][(df[str(sel_idx[i])]==0)],df[str(y2_idx[i])][(df[str(sel_idx[i])]==0)],s=1,alpha=0.2,color="blue")
        #plt.scatter(y1_eq1,y2_eq1,s=1,alpha=1,color="green")
        plt.scatter(current_df[str(y1_idx[i])],current_df[str(y2_idx[i])],s=1,alpha=1,color="green",label="y_hat_1!=y_hat_2")
        plt.scatter(same_df[[str(y1_idx[i])]],same_df[[str(y2_idx[i])]],s=1,alpha=1,color="red",label="y_hat_1==y_hat_2")
        #plt.axvline(x=-np.log(0.5),color="black")
        #plt.axhline(y=-np.log(0.5),color="black")
        #plt.axvline(x=-np.log(0.1),color="green")
        #plt.axhline(y=-np.log(0.1),color="green")
        plt.xlabel("-log(y_hat_1)")
        plt.ylabel("-log(y_hat_2)")
        plt.title("-log(y_hat) vs -log(y_hat) at epoch "+str(epochs[i]))
        #log x axis
        plt.xscale("log")
        plt.yscale("log")
        #axis limits
        plt.xlim(10e-5,10e1) #(10e-10,10e5)
        plt.ylim(10e-5,10e1) #(10e-8,10e1)
        #plot legend middle left
        plt.legend(loc="center left")

        #add grid lines
        plt.grid()
        plt.savefig(str(output_name)+str(epochs[i])+".png")
        #clear plot
        plt.clf()

def FitLines(file_name,output_name):
    df = pd.read_csv(file_name)
    #df2 = pd.read_csv("RandomCorrectLossFIM.csv")
    fim_idx =    [1,5,9,13,17]
    fimsel_idx = [2,6,10,14,18]
    y_idx =      [3,7,11,15,19]
    ysel_idx =  [4,8,12,16,20]

    epochs = [0,20,40,60,80]
    x = np.logspace(-10,5,1000)
    #y = -a*log(b*x)+c
    a1=800
    b1=2
    c1=0
    c2=0.0001
    y = -a1*np.log(1-np.exp(-b1*x))
    #y2 = -np.log(1-x)
    for i in range(len(fim_idx)):
        #get the losses and fim values when corr is 1
        #corr1_losses = df[str(loss_idx[i])][df[str(corr_idx[i])]==1]
        fimsel=[1,0,1,0]
        ysel=[1,1,0,0]
        colors=["green","blue","orange","red"]
        for j in range(len(fimsel)):
            plt.scatter(-df[str(y_idx[i])][(df[str(fimsel_idx[i])]==fimsel[j])&(df[str(ysel_idx[i])]==ysel[j])],df[str(fim_idx[i])][(df[str(fimsel_idx[i])]==fimsel[j])&(df[str(ysel_idx[i])]==ysel[j])],s=1,alpha=0.2,color=colors[j],label="FIM L="+str(fimsel[j])+", y_hat L="+str(ysel[j]))
        
        plt.plot(x,y,label="y=-log(x)")
        #plt.plot(x,y2,label="y=-log(1-x)")
        plt.ylabel("FIM")
        plt.xlabel("-log(y_hat)")
        plt.title("FIM vs -log(y_hat) at epoch "+str(epochs[i]))
        #log x axis
        plt.xscale("log")
        plt.yscale("log")
        #axis limits
        plt.ylim(10e-10,10e5) #(10e-10,10e5)
        plt.xlim(10e-8,10e1) #(10e-8,10e1)
        #plot legend middle left
        plt.legend(loc="center left")

        #add grid lines
        plt.grid()
        plt.savefig(str(output_name)+str(epochs[i])+".png")
        #clear plot
        plt.clf()

def YhatFIM(file_name,output_name):
    df = pd.read_csv(file_name) #FIM, log(y_hat)
    #df2 = pd.read_csv("RandomCorrectLossFIM.csv")
    y1_idx =    [1,3,5,7,9] #FIM
    y2_idx =    [2,4,6,8,10] #log(y_hat)

    epochs = [0,20,40,60,80]
    for i in range(len(y1_idx)):
        current_df =df[[str(y1_idx[i]),str(y2_idx[i])]]
        #make the log(y_hat) values positive with a negative sign
        current_df[str(y2_idx[i])] = (current_df[str(y2_idx[i])])
        #current_df[str(y1_idx[i])] = np.log(current_df[str(y1_idx[i])])
        #remove the nans
        current_df = current_df.dropna()

        #make a df where -log(y_hat) is less than 1
        reduced_df = current_df[(current_df[str(y2_idx[i])]<1)]
        #All pair represent the same y_hat output

        def func(x,a,b):
            return  a*np.log(1+x)**2
        def func2(x,a,b):
            return  np.exp(a*x-1)
        #popt2, pcov2 = curve_fit(func, reduced_df[str(y2_idx[i])], reduced_df[str(y1_idx[i])])
        popt, pcov = curve_fit(func, -current_df[str(y2_idx[i])], current_df[str(y1_idx[i])])
        print(popt)

        #x = np.linspace(0.0001,0.9999,1000)
        x = np.logspace(-8,2,1000)
        #plt.plot(x,func(x,3000,1),label="y=-log(1-e^(-x))",color="red")
        #plt.plot(x,func2(x,50,0),label="y=-log(1-e^(-x))",color="blue")
        #plt.plot(x,func(x,*popt),label="y=-log(1-e^(-x))")
        

        #plt.plot(x,y,label="y=-log(1-e^(-x))",alpha=0.5)
        #plt.plot(x,x,label="y=x",alpha=0.5,color="black")

        plt.scatter(-current_df[str(y2_idx[i])],current_df[str(y1_idx[i])]-[func(x,*popt) for x in -current_df[str(y2_idx[i])]],s=1,alpha=1,color="green",label="y_hat_1!=y_hat_2")
        plt.scatter(-reduced_df[str(y2_idx[i])],reduced_df[str(y1_idx[i])]-[func(x,*popt) for x in -current_df[str(y2_idx[i])]],s=1,alpha=1,color="red",label="y_hat_1==y_hat_2")
        #plt.axvline(x=-np.log(0.5),color="black")
        #plt.axhline(y=-np.log(0.5),color="black")
        #plt.axvline(x=-np.log(0.1),color="green")
        #plt.axhline(y=-np.log(0.1),color="green")
        plt.xlabel("-log(y_hat)")
        plt.ylabel("FIM(y_hat)")
        plt.title("-log(y_hat) vs FIM(log(y_hat)) at epoch "+str(epochs[i]))
        #log x axis
        plt.xscale("log")
        #plt.yscale("log")
        #axis limits
        #plt.xlim(10e-11,10e5) #(10e-10,10e5)
        #plt.ylim(10e-11,10e5) #(10e-8,10e1)
        #plot legend middle left
        #plt.legend(loc="center left")

        #add grid lines
        plt.grid()
        plt.savefig(str(output_name)+str(epochs[i])+".png")
        #clear plot
        plt.clf()

def LayerFIM(file_name,output_name):
    df = pd.read_csv(file_name) #FIM, log(y_hat)
    #df2 = pd.read_csv("RandomCorrectLossFIM.csv")
    l_idx =         [[1,10],[13,22]]
    loss_idx =      [11,23]
    o_idx =         [12,24]

    def col_select(x):
        #x is between 0 and 1
        # return an rgb color from red to blue (red =1) (blue = 0)
        return (x,0,1-x)
    # 10 colors from red to green
    colors = np.linspace(0,1,10)

    epochs = [0,20]
    for i in range(len(epochs)):
        for j in range(l_idx[i][0],l_idx[i][1]):
            if j %2 != 0:
                plt.scatter(df[str(j)],np.exp(df[str(o_idx[i])]),s=1,alpha=0.2,color=col_select(colors[j-l_idx[i][0]]),label="Layer "+str(j-l_idx[i][0]))
        #plt.axvline(x=-np.log(0.5),color="black")
        #plt.axhline(y=-np.log(0.5),color="black")
        #plt.axvline(x=-np.log(0.1),color="green")
        #plt.axhline(y=-np.log(0.1),color="green")
        plt.xlabel("FIM")
        plt.ylabel("y_hat")
        plt.title("FIM(log(y_hat)) vs y_hat at epoch "+str(epochs[i]))
        #log x axis
        plt.xscale("log")
        #plt.yscale("log")
        #axis limits
        #plt.xlim(0,30) #(10e-10,10e5)
        #plt.ylim(0,10e5) #(10e-8,10e1)
        #plot legend middle left
        plt.legend(loc="upper left")

        #add grid lines
        plt.grid()
        plt.savefig(str(output_name)+str(epochs[i])+".png")
        #clear plot
        plt.clf()

def alphaTime(file_name,output_name):
    df = pd.read_csv(file_name) #alpha
    df = df.drop(columns=["Unnamed: 0","0"])
    dft = pd.read_csv("TestLayerAlphaLossFIM.csv")
    dft = dft.drop(columns=["Unnamed: 0","0"])
    #get the values of alpha in array every other value
    colors = ["red","blue","green","orange","purple","brown","pink","grey","black","yellow"]
    c = 0
    for i in range(10):
        if i % 2 == 0:
            plt.plot(df.iloc[i].values,label="Layer "+str(i),color=colors[c])
            plt.plot(dft.iloc[i].values,linestyle="--",color=colors[c])
            c+=1



    #plt.plot(alphas,label="Alpha")
    #plt.plot(var,label="Variance")
    plt.xlabel("Epoch")
    plt.ylabel("Alpha")
    plt.yscale("log")
    plt.title("Layer Alpha vs Epoch (Train & Test)")
    plt.grid()
    plt.legend(loc="upper left")
    plt.savefig(str(output_name)+".png")

def alphaPlus(train_file_names,test_file_names,output_name):
    dfs = []
    for name in train_file_names:
        dfs.append(pd.read_csv(name))
    
    for df in dfs:
        print(df.head())
    
    #test_dfs = []
    #for name in test_file_names:
    #    test_dfs.append(pd.read_csv(name))

    F = np.arange(1,40,2)
    y = np.arange(2,41,2)
    #calc the alpha for each epoch
    def func(x,a):
        return a*np.log(1+x)**2

    #alpha values for each epoch for each df
    
    alpha_list = np.empty((len(train_file_names), 20))
    #alpha_test_list = np.empty((len(train_file_names), 20))

    print(alpha_list.shape)
    print(len(dfs),len(F))

    F_avg = []
    y_avg = []
    j = 0
    for df in dfs:
        alpha = [] #[[t0,t1,t2,t3,t4...],[t0,t1,t2,t3,t4...]...]
        for i in range(len(F)):
            series = df[[str(y[i]),str(F[i])]]
            series = series.dropna()
            if len(series) == 0:
                alpha.append(0)
                continue
            popt, pcov = curve_fit(func, -series[str(y[i])],series[str(F[i])])
            alpha.append(popt[0])
        
        print(len(alpha))
        print(j)
        alpha_list[j] = alpha
        j+=1
    j = 0
    # for df in test_dfs:
    #     alpha_test = [] #[[t0,t1,t2,t3,t4...],[t0,t1,t2,t3,t4...]...]
    #     for i in range(len(F)):
    #         series = df[[str(y[i]),str(F[i])]]
    #         series = series.dropna()
    #         popt, pcov = curve_fit(func, -series[str(y[i])],series[str(F[i])])
    #         alpha_test.append(popt[0])

        
    #     alpha_test_list[j] = alpha_test
    #     j+=1
    
    def first_nan_index(df):
        return df[df.isna()].index[0] if df.isna().any() else None
    
    def first_nan_column(df):
        for column in df.columns:
            if df[column].isna().any():
                return column
        return None
    
    nan_index = []
    for df in dfs:
        #find the first value index that is a nan
        nan_index.append(first_nan_column(df.drop(columns=df.columns[[0,1]])))
    
    print(nan_index)

    #F_avg.append(np.mean(series[str(F[i])]))
    #y_avg.append(np.mean(-series[str(y[i])]))

    BS = [32,32,32,32,32,32]
    params = ["0.1","0.05","0.01","0.005","0.001","0.0001","Lin(0.001-0.01)","lin(0.01-0.0001)","LRStep[(0,0.005)-(5,0.001)-(30,0.01)]","LRCSR[0.01,e*10,2,1,0]"]

    colours = ["red","blue","green","orange","purple","brown","pink","grey","black","yellow"]
    #colors = np.linspace(0,1,len(alpha_list))
    for i in range(len(alpha_list)):
        steps = np.arange(1,100,5)
        plt.plot(steps,alpha_list[i],label=str(params[i]),color=colours[i])
        #plt.plot(steps,alpha_test_list[i],label=str(params[i]),color=(colors[i],0,1-colors[i]),linestyle="--")
        if nan_index[i] != None:
            plt.axvline(x=int(int(nan_index[i])/2)*5,color=colours[i],linestyle=":",alpha=0.25)
    

    plt.xlabel("Epoch")
    plt.ylabel("Alpha")
    plt.xlim(0,100)
    #plt.ylim(0,14000)
    #plt.yscale("log")
    plt.title("Alpha vs Epoch (LR)")
    #plt.legend(loc="upper left",bbox_to_anchor=(1,1))
    plt.grid(alpha=0.5)
    plt.savefig(str(output_name)+"EpochNoLeg.png",bbox_inches='tight')
    
    plt.clf()
    plnt()


    #plot the loss vs the alpha
    loss_df = pd.read_csv("BSLosses.csv")
    #only keep every 5th row
    loss_df = loss_df.iloc[::5]
    loss_list = np.array([loss_df['BS32 - loss'].values,loss_df['BS8 - loss'].values,loss_df['BS64 - loss'].values])
    loss_test_list = np.array([loss_df['BS32 - val_loss'].values,loss_df['BS8 - val_loss'].values,loss_df['BS64 - val_loss'].values])
    print(loss_list.shape)

    
    print(loss_df.head())
    print(loss_df.shape)
    for i in range(len(alpha_list)):
        plt.plot(alpha_list[i],loss_list[i],label="BS="+str(BS[i]),color=colours[i],marker='s',markersize=5)
        plt.plot(alpha_test_list[i],loss_test_list[i],label="BS="+str(BS[i]),color=colours[i],linestyle="--",marker='s',markersize=5)
        #plt.axvline(x=int(int(nan_index[i])/2),color=colours[i],linestyle=":")

    plt.xlabel("Alpha")
    plt.ylabel("Loss")
    #plt.yscale("log")
    plt.xscale("log")
    plt.title("Loss vs Alpha (Batch Size)")
    plt.legend()
    
    plt.savefig(str(output_name)+"LossAlpha.png")
    plt.clf()


    pnt()


    #colour list from red to blue
    colors = np.linspace(0,1,len(F))

    for i in range(len(F)):
        plt.plot(np.logspace(-3,1,1000),func(np.logspace(-3,1,1000),alpha[i]),label="Epoch "+str(F[i]),color=(colors[i],0,1-colors[i]))
    for i in range(len(F)):
        plt.plot(y_avg[i],F_avg[i],linestyle="--",marker='s',color=(colors[i],0,1-colors[i]))
    #for i in range(len(y_avg)-1):
    #    plt.annotate('', xy=(y_avg[i+1], F_avg[i+1]), xytext=(y_avg[i], F_avg[i]),
    #             arrowprops=dict(facecolor='black',alpha=0.5, shrink=1,headwidth=4, headlength=2))
    plt.xlabel("-log(y_hat)")
    plt.ylabel("FIM")
    plt.yscale("log")
    plt.xscale("log")
    #plt.legend()
    plt.title("FIM vs -log(y_hat) with alpha fit & averages over time")
    plt.savefig(str(output_name)+"Fit.png")
    plt.clf()

    #plot a single epochs F vs y_hat
    i = 18
    plt.scatter(-df[str(y[i])],df[str(F[i])],s=1,alpha=0.2,color="red")
    plt.scatter(-df2[str(y[i])],df2[str(F[i])],s=1,alpha=0.2,color="blue")
    plt.scatter(y_avg[i],F_avg[i],s=20,alpha=1,color="black")
    print(y_avg[i],F_avg[i])
    plt.xlabel("-log(y_hat)")
    plt.ylabel("FIM")
    plt.yscale("log")
    plt.xscale("log")
    plt.title("FIM vs -log(y_hat) at epoch "+str((y[i]/2)*5))
    plt.savefig(str(output_name)+"Scatter"+str((y[i]/2)*5)+".png")


def FIMtypes(train_file_names,output_name):
    dfs = []
    for path in train_file_names:
        dfs.append(pd.read_csv(path))
    
    for df in dfs:
        print(df.head())
    
    #test_dfs = []
    #for name in test_file_names:
    #    test_dfs.append(pd.read_csv(name))

    F = np.arange(1,40,2)
    y = np.arange(2,41,2)
    #calc the alpha for each epoch
    def func(x,a):
        return a*np.log(1+x)**2

    #alpha values for each epoch for each df
    alpha_list = np.empty((len(train_file_names), 20))

    F_avg = []
    y_avg = []
    j = 0
    for df in dfs:
        alpha = [] #[[t0,t1,t2,t3,t4...],[t0,t1,t2,t3,t4...]...]
        for i in range(len(F)):
            series = df[[str(y[i]),str(F[i])]]
            series = series.dropna()
            if len(series) == 0:
                alpha.append(0)
                continue
            popt, pcov = curve_fit(func, -series[str(y[i])],series[str(F[i])])
            alpha.append(popt[0])
        
        print(len(alpha))
        print(j)
        alpha_list[j] = alpha
        j+=1
    j = 0

    
    def first_nan_index(df):
        return df[df.isna()].index[0] if df.isna().any() else None
    
    def first_nan_column(df):
        for column in df.columns:
            if df[column].isna().any():
                return column
        return None
    
    nan_index = []
    for df in dfs:
        #find the first value index that is a nan
        nan_index.append(first_nan_column(df.drop(columns=df.columns[[0,1]])))
    
    print(nan_index)
    params = ["Train","Test","0.25","0.5","0.75","Random"]
    colours = ["red","blue","green","orange","purple","brown","pink","grey","black","yellow"]
    #colors = np.linspace(0,1,len(alpha_list))
    #plot the -y_hat vs FIM for a given epoch and the alpha line

    steps = np.arange(1,100,5)
    params = ["Stat","Emp","Flat"]
    for i, df in enumerate(dfs):
        plt.scatter(-df[str(y[4])],df[str(F[4])],s=1,alpha=0.4,color=colours[i],label=params[i])
    
    plt.xlabel("-y_hat")
    plt.ylabel("FIM")
    plt.yscale("log")
    plt.xscale("log")
    plt.title("-y_hat vs FIM (Agg Type) Epoch 20")
    plt.legend()
    plt.savefig(str(output_name)+"Epoch20.png")
    plt.clf()
    for i, df in enumerate(dfs):
        plt.scatter(-df[str(y[0])],df[str(F[0])],s=1,alpha=0.4,color=colours[i],label=params[i])
    
    plt.xlabel("-y_hat")
    plt.ylabel("FIM")
    plt.yscale("log")
    plt.xscale("log")
    plt.title("-y_hat vs FIM (Agg Type) Epoch 0")
    plt.legend()
    plt.savefig(str(output_name)+"Epoch0.png")
    plt.clf()
    plnt()
    
    
def accgraphs(file_name,file_names,output_name):
    df = pd.read_csv(file_name)
    #set index to step
    df = df.set_index("Step")
    #find the max of each col
    maxes_idx = df.idxmax()
    #find the max of each col
    maxes = df.max()
    #apply the same colours as the alpha plot
    info = [("LogOuts(Train/Test)FIMSampleDistLR0_1 - val_categorical_accuracy","red","0.1"),
        ("LogOuts(Train/Test)FIMSampleDistLR0_05 - val_categorical_accuracy", "blue","0.05"),
        ("LogOuts(Train/Test)FIMSampleDist - val_categorical_accuracy","green","0.01"),
        ("LogOuts(Train/Test)FIMSampleDistLR0_005 - val_categorical_accuracy","orange","0.005"),
        ("LogOuts(Train/Test)FIMSampleDistLR0_001 - val_categorical_accuracy","purple","0.001"),
        ("LogOuts(Train/Test)FIMSampleDistLR0_0001 - val_categorical_accuracy","brown","0.0001"),
        ("LRLinear[(0,0.001)-(100,0.01)] - val_categorical_accuracy","pink","lin(0.001-0.01)"),
        ("LRLinear[(0,0.01)-(100,0.0001)] - val_categorical_accuracy","grey","lin(0.01-0.0001)"),
        ("LRStep[(0,0.005)-(5,0.001)-(30,0.01)] - val_categorical_accuracy","black","step[(0,0.005)-(5,0.001)-(30,0.01)]"),
        ("LRCSR[0.01,e*10,2,1,0] - val_categorical_accuracy","yellow","CSR[0.01,e*10,2,1,0)")]
    
    info_df = pd.DataFrame({"Name":[i[2] for i in info],
                            "Colour":[i[1] for i in info],
                            "idx":[i[0] for i in info],
                            "file_name":file_names,
                            })
    info_df = info_df.set_index("idx",drop=False)
    info_df['max'] = maxes
    info_df['max_idx'] = maxes_idx
    info_df['last_TA'] = df.iloc[-1]
    info_df = info_df.set_index("Name",drop=False)

    #sort via the max column
    info_df = info_df.sort_values(by="max",ascending=False)
    
    # #loop through the maxes and plot the accuracy based on the info
    # for row in info_df.iterrows():
    #     plt.bar(row[1][0],row[1][2],label=row[1][0],color=row[1][1])

    #     #plt.bar(i[2],maxes[i[0]],label=i[2],color=i[1])
    # plt.xlabel("Learning Rate")
    # plt.ylabel("Max Test Accuracy")
    # #rotate the x labels
    # plt.xticks(rotation=45)
    # plt.title("Max Test Accuracy vs Learning Rate")
    # plt.ylim(0.4,0.75)
    # plt.savefig(str(output_name)+".png")
    # plt.clf()

    pnt()

    #Figure plotting alpha vs test accuracy
    dfs = []
    for name in file_names:
        dfs.append(pd.read_csv(name))

    F = np.arange(1,40,2)
    y = np.arange(2,41,2)
    #calc the alpha for each epoch
    def func(x,a):
        return a*np.log(1+x)**2

    #alpha values for each epoch for each df
    alpha_list = np.empty((len(file_names), 20))

    F_avg = []
    y_avg = []
    alpha_file_names = [i for i in file_names]
    j = 0
    for i_df in dfs:
        alpha = [] #[[t0,t1,t2,t3,t4...],[t0,t1,t2,t3,t4...]...]
        for i in range(len(F)):
            series = i_df[[str(y[i]),str(F[i])]]
            series = series.dropna()
            if len(series) == 0:
                alpha.append(0)
                continue
            popt, pcov = curve_fit(func, -series[str(y[i])],series[str(F[i])])
            alpha.append(popt[0])
        
        alpha_list[j] = alpha
        j+=1
    #print(alpha_list)
    print(alpha_file_names)

    info_df = info_df.set_index("file_name",drop=False)
    print(info_df.columns)

    #get the alpha at the point of max accuracy
    alpha_at_max_TA = []
    for a,fn in zip(alpha_list,alpha_file_names):
        alpha_at_max_TA.append(a[int(info_df.max_idx.loc[fn]/5)])
    info_df = info_df.set_index("Name",drop=True)
    
    
    FIM_df = pd.DataFrame({"file_name_og":file_names,
                            "Name":[i[2] for i in info],
                            #"idx":[i[0] for i in info],
                            "max_alpha":alpha_list.max(axis=1),
                            "fin_alpha":alpha_list[:,-1],
                            "alpha_at_max_TA":alpha_at_max_TA})
    
    FIM_df = FIM_df.set_index("Name",drop=False)
    print(FIM_df.columns)
    #join the info_df and the FIM_df
    FIM_df = FIM_df.join(info_df)
    FIM_df = FIM_df.set_index("file_name",drop=True)
    print(FIM_df)
    plt.clf()

    #plot the max alpha vs the max accuracy
    for i in range(len(FIM_df)):
        plt.scatter(FIM_df['max_alpha'].iloc[i],FIM_df['last_TA'].iloc[i],label=FIM_df['Name'].iloc[i],color=FIM_df['Colour'].iloc[i],marker='o',facecolors='none')
        plt.scatter(FIM_df['alpha_at_max_TA'].iloc[i],FIM_df['max'].iloc[i],color=FIM_df['Colour'].iloc[i],marker='x')

    plt.xlabel("Alpha")
    plt.ylabel("Test Accuracy")
    #plt.xscale("log")
    #plt.yscale("log")
    plt.grid(alpha=0.5)
    plt.title("Max Accuracy vs Alpha")
    plt.legend(loc="lower right")
    plt.savefig("AlphaMaxAcc.png",bbox_inches='tight')
    plt.clf()
    
def lr_testacc(acc_file,output_name):
    df = pd.read_csv(acc_file)
    #set index to step
    df = df.set_index("Step")
    #find the max of each col
    selected =['LogOuts(Train/Test)FIMSampleDistLR0_005 - val_categorical_accuracy',
       'LogOuts(Train/Test)FIMSampleDistLR0_05 - val_categorical_accuracy',
       'LogOuts(Train/Test)FIMSampleDistLR0_0001 - val_categorical_accuracy',
       'LogOuts(Train/Test)FIMSampleDistLR0_001 - val_categorical_accuracy',
       'LogOuts(Train/Test)FIMSampleDistLR0_1 - val_categorical_accuracy',
       'LogOuts(Train/Test)FIMSampleDist - val_categorical_accuracy']
    maxes = df[selected].max()
    lrs = [0.005,0.05,0.0001,0.001,0.1,0.01]
    classes = [1,0,2,2,0,1]
    cols = ["green","orange"]

    
    print(df.columns)
    df = pd.DataFrame({"LR":lrs,"Max":maxes,"Class":classes})
    df = df.set_index("LR",drop=False)
    df = df.sort_index()
    print(df)

    plt.plot(df['LR'],df['Max'],color="blue")
    transitions = [(0.05+0.01)/2,(0.005+0.001)/2]
    for i in range(len(transitions)):
        plt.axvline(x=transitions[i],color=cols[i],linestyle="--")



    #plt.plot(df['LR'],df['Max'])
    plt.xlabel("Learning Rate")
    plt.ylabel("Max Test Accuracy")
    plt.xscale("log")
    plt.title("Max Test Accuracy vs Learning Rate")
    plt.legend()
    plt.savefig("LRVsTA.png")
    plt.clf()
    


    

if __name__ == "__main__":
    #corrGraphs("NormalSelectedLossFIM.csv","NormalSelectedLossFIMNotCorr")
    #batch_csv("NormalBatch80LossFIM.csv","NormalBatch80LossFIM")
    #loss_fim_gap("DenseNormalLossFIM.csv")
    #csv2_to_graphs("MagNormalLossFIM.csv","MagNormal")
    #alpha_csv("AugmentTradNoiseLossFIM.csv","AugmentTradNoiseLossFIM")

    #csv_to_graphs("CCEoff-0_0001LossFIM.csv","CCEoff-0_0001LossFIM")

    #DisjointGraphs("DisjointFIMandYhatLossFIM.csv","DisjointFIMandYhat")
    #FitLines("DisjointFIMandYhatLossFIM.csv","Fitlines")
    #YhatYhat("Yhats2ClassLossFIM.csv","negLogYhatToLogYhat2Class")
    #YhatFIM("FIMSampledLogOutFIM.csv","-log(y_hat)FIMFitNormed")
    #LayerFIM("LayerLossFIM.csv","LayerLossFIM")
    alphaPlus(["OptADAMLR0_0001LogOutFIM.csv",
                "LRLinear[(0,0.01)-(100,0.0001)LogOutFIM.csv",
       ],
       None,
       "AlphaFlatDistOpt")
    #alphaPlus("LayerAlphaLossFIM.csv","LayerBiasAlphaPlus")
    #FIMtypes(["typeStatLogOutFIM.csv",
        # "typeEmpLogOutFIM.csv",
        # "typeFlatLogOutFIM.csv"],
        # "FIMTypes")
    # accgraphs("LRTestAcc.csv",
    #     ["FIMFlatDistLR0_1LogOutFIM.csv",
    #     "FIMFlatDistLR0_05LogOutFIM.csv",
    #     "FIMFlatDistLogOutFIM.csv",
    #     "FIMFlatDistLR0_005LogOutFIM.csv",
    #     "FIMFlatDistLR0_001LogOutFIM.csv",
    #     "FIMFlatDistLR0_0001LogOutFIM.csv",
    #     "LRLinear[(0,0.001)-(100,0.01)]LogOutFIM.csv",
    #     "LRLinear[(0,0.01)-(100,0.0001)LogOutFIM.csv",
    #     "LRStep[(0,0.005)-(5,0.001)-(30,0.01)]LogOutFIM.csv",
    #     "LRCSR[0.01,e*10,2,1,0]LogOutFIM.csv"],
    #     "LRTestAcc")
    #lr_testacc("LRTestAcc.csv","CIFAR-")



#Experiment list
#FIM sampled based on output dist unless otherwise stated.
#Model 5 layer cnn unless stated otheriwse.

#1.  NormalSelectedTestLossFIM.csv   FIM, loss, bool(FIM calc uses correct ouput class or not)
#2.  AllOutputsNormal                FIM, loss for one image. FIM calced for all output classes. one epoch.
#3.  AugmentTradFIMLossFIM           FIM, loss. Augmented data with noise and y flip.
#4.  AugmentTradNoiseLossFIM         FIM, loss. Augmented data with noise.
#5.  AugmentTradYFlipLossFIM         FIM, loss. Augmented data with y flip.
#6.  DenseFIMvsLoss                  FIM, loss. Dense network.
#7.  LargeFIMvsLoss                  FIM, loss. Large CNN network.
#8.  LBSFIMvsLoss                    FIM, loss. Large batch size.
#9.  MagNormal                       FIM, loss, grad norm. Magnitude of the gradient.
#10. MixFIMGap                       Attempt to find the K value with the Gap between loss and FIM.
#11. MisslabelledFIMLoss             FIM, loss. Misslabelled data 20%. Red is misslabelled.
#12. MNISTFIMLoss                    FIM, loss. MNIST data.
#13. NormalAccLossFIM                FIM, loss, accuracy. Green is correctly classified (highest y_hat=label). Red is missclassified.
#14. NormalBatchLossFIM              FIM, loss. A single image across a single epoch.
#15. NormalFIMLoss                   FIM, loss. Normal data.
#16. NormalSelectedLossFIM           FIM, loss, bool. Green if FIM is calculated using correct output class. Red if not.
#17. NormalSelectedTestLossFIM       FIM, loss, bool. Test data used. Green if FIM is calculated using correct output class. Red if not.
#18. RandomFIMvsLoss                 FIM, loss. Green is dataset. Red is random data. Trained on both.
#19. SBSFIMvsLoss                    FIM, loss. Small batch size.
#20. SmallFIMvsLoss                  FIM, loss. Small CNN network.
#21. TestNormal                      FIM, loss. Test data used.
#21. 2classesLossFIM                 FIM, loss. 2 classes from CIFAR10
#22. CCEoff-0_00001LossFIM           FIM, loss. CCE loss with -0.00001 as one hot off value.
#23. CCEoff0_0001LossFIM             FIM, loss. CCE loss with 0.0001 as one hot off value.
#24. FIMFlatDistLogOutFIM            FIM, log y_hat. Both from y_hat used for FIM calc. Uniform distribution to select y_hat.
#25. FIMSampledLogOutFIM             FIM, log y_hat. Both from y_hat used for FIM calc. Sampled from output distribution to select y_hat.
#26. DisjointFIMandYHat              FIM, is fim calced with label class, log(y_hat), is y_hat using the label class. Uniform dist.
#27. YhatsToFIMSimulated             FIM, -log(y_hat). y_hat is created from a uniform distribution between 0 and 1.
#28. LayerLossFIM                    FIM per layer, loss, output used. 