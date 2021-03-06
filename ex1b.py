import numpy as np
import matplotlib.pyplot as plt


# ==========================  1(b)   ==========================
from ex1functions import BoxMuller,GaussianPDF

#perform Box Muller tranform from these uniform variates to a standard normal variate
# This function draws two lists of uniform variates from within
# (transformed again to given mean=3.0 and stdev=2.4)
mean=3.0
stddev=2.4
normdraw=BoxMuller(N=1000,mean=mean,stdv=stddev)

#plot the result
freq,bins,_=plt.hist(normdraw,bins=20,edgecolor='black', linewidth=1.5,density=True)
#list of stddev locations on x-axis
sigmabars=np.concatenate((mean-np.linspace(5*stddev,stddev,5),mean+np.linspace(stddev,5*stddev,5)))
#plot stddev locations and mean
plt.vlines(mean,0,GaussianPDF(mean,mu=mean,s=stddev),linewidth=2,linestyles=':')
plt.vlines(sigmabars,0,GaussianPDF(sigmabars,mu=mean,s=stddev),linewidth=1.8,linestyles=':',color='r')
plt.title("Box-Muller Normal Variate ($n_{draws}$ = 1,000,000)")
plt.xlabel("True values")
plt.ylabel("Pseudorandom (pdf-normalized) Frequency")
#plot expected shape of ideal normal
plt.plot(np.linspace(bins[0]-2,bins[-1]+2,100), GaussianPDF(np.linspace(bins[0]-2,bins[-1]+2,100),mu=mean,s=stddev),linewidth=2.2,linestyle="--",color='k',label='Ideal Normal ($\mu$='+str(mean)+', $\sigma$='+str(stddev)+')')
plt.legend()

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig("./plots/1b_gaussiancomparison.png")
plt.clf()
