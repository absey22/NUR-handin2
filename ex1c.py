import numpy as np
import matplotlib.pyplot as plt


# ==========================  1(c)   ==========================
from ex1functions import mykstest,BoxMuller
from scipy.stats import kstest

#Prove: null hypothesis is that actual data follows a normal distribution
# accomplish by finding max distance between Gaussian CDF and data CDF

#The KS statistic D can be calculated as shown in Section 6.14 of Press et al.
# a p-value of the signficance of the D of this observation can therefore be found
# from that distribution.

#is the probability that a value as large as D would occur if data was drawn from cdf.

#If the p-value is greater than the significance level (say 5%), then we cannot reject the hypothesis that the data come from the given distribution.

N=int(1e5) # number of points to reach up to in testing
data=BoxMuller(N,mean=0,stdv=1.0) # make random realization of data drawn from my Box-Muller algorithm (normal variate from 2 uniform variates)

dex=0.1 # to increment by ("n_points=10^dex")
start,stop=1,5 # to start at and stop at (stop=log_10(N))
cnt=np.ones(int((stop-start)/dex)+1) # +1 to reach up to N in slicing randomdata
pval=np.zeros((len(cnt),2)) # 2 for accepting a D AND a p-value
Dval=pval.copy() # the same data shape will be required to store the D statistic

for i in range(len(cnt)):
    cnt[i]+=round(i*dex,2)   # increment the cnt by dex each loop for slicing & plotting
    #normaldraw=BoxMuller(10**cnt[i],mean=0,stdv=1.0) #create a realization of data drawn from my Box-Muller algorithm (normal variates)
    dataslice=data[:int(10**cnt[i])]  # sucessively larger slices into randomdata to analyze
    Dval[i,0],pval[i,0]=mykstest(dataslice) # take D,pval from mykstest() (default: normal cdf)
    Dval[i,1],pval[i,1]=kstest(dataslice,'norm') # take D,pval from scipy.stats.kstest


plt.subplot(2,1,1)
plt.title("K-S Test: Box-Muller Normal rvs")
plt.plot(cnt,pval[:,0],label="My p-value")
plt.plot(cnt,pval[:,1],":",label="SciPy p-value")
plt.ylabel("p-value")
plt.yscale("log")
plt.ylim((10**-3,10**0.2))
#plt.ylim((10**-2.8,10**1.1))
plt.hlines(0.05,0.8,5.2,linewidth=0.8,linestyles=':')
plt.text(4.35,10**-1.6,'2-$\sigma$ rejection',color='r')
plt.hlines(0.003,0.8,5.2, linewidth=0.8,linestyles='--')
plt.text(4.35,10**-2.82,'3-$\sigma$ rejection',color='r')
plt.legend()

plt.subplot(2,1,2)
plt.plot(cnt,Dval[:,0],label="My D statistic")
plt.plot(cnt,Dval[:,1],":",label="SciPy D statistic")
plt.xlabel("$log_{10}(N_{points})$")
plt.ylabel("D statistic")
plt.yscale("log")
plt.ylim(top=(10**0.2))
plt.legend()

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig("./plots/1c_kstest.png")
plt.clf()
