import numpy as np
import matplotlib.pyplot as plt


# ==========================  1(e)   ==========================
from ex1functions import kuiperstest,GaussianPDF

#Kuiper test will be robust against cyclic data with a phase in the x-direction

dataset=np.genfromtxt("randomnumbers.txt")



for j in range(dataset.shape[1]):
    dex=0.1 
    start,stop=1,5 
    cnt=np.ones(int((stop-start)/dex)+1)
    mykuiper=np.zeros((len(cnt),3))

    data=dataset[:,j] # FOR EACH COLUMN in dataset:: take one of the data sets
    for i in range(len(cnt)): #slice it incrementally
        cnt[i]+=round(i*dex,2)
        dataslice=data[:int(10**cnt[i])]
        mykuiper[i]=kuiperstest(dataslice) # get D+,D-,pval from my Kuiper test

    
    plt.subplot(3,1,1)
    plt.title("Kuiper Test: 'randomnumbers.txt' Dataset #"+str(j+1))
    plt.hist(data,bins=20,edgecolor='black', linewidth=1.5,density=True)
    plt.plot(np.linspace(-5,5,55),GaussianPDF(np.linspace(-5,5,55)),linewidth=2,linestyle=":",color='k')
    plt.xlabel("True Values")
    plt.ylabel("normlzd. rv freq.")
    plt.xlim((-5,5))
    plt.ylim((0,0.45))

    plt.subplot(3,1,2)
    plt.plot(cnt,mykuiper[:,2],'r',label="My Kuiper p-value")
    plt.xlabel("$log_{10}(N_{points})$")
    plt.ylabel("p-value")
    plt.hlines(0.05,0.8,5.2,linewidth=0.8,linestyles=':')
    plt.text(4.35,10**-1.8,'2-$\sigma$ rejection',fontsize=10,color='r')
    plt.hlines(0.003,0.8,5.2, linewidth=0.8,linestyles='--')
    plt.text(4.35,10**-3.2,'3-$\sigma$ rejection',fontsize=10,color='r')
    plt.hlines(0.0001,0.8,5.2, linewidth=0.8,linestyles='-')
    plt.text(4.35,10**-4.6,'4-$\sigma$ rejection',color='r')
    plt.yscale("log")
    plt.ylim((10**-5.5,10**0.5))
    plt.legend(loc=3)

    plt.subplot(3,1,3)
    plt.plot(cnt,np.add(mykuiper[:,0],mykuiper[:,1]),label="My Kuiper V statistic")
    plt.xlabel("$log_{10}(N_{points})$")
    plt.ylabel("V statistic")
    plt.yscale("log")
    plt.ylim((10**-2.5,10**0.5))
    plt.legend(loc=3)


    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig("./plots/1e_kuipertest_data"+str(j+1)+".png")
    plt.clf()
