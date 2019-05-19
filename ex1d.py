import numpy as np
import matplotlib.pyplot as plt


# ==========================  1(d)   ==========================
from ex1functions import kuiperstest,BoxMuller, GaussianCDF
#from astropy.stats import kuiper
from scipy.stats import kstest

#Kuiper V = D+ + D- is a statistic that is invariant under all shifts
# and parametrizations on a circle created by wrapping around the x-axis.

#from kuiper source code:
"""
    Stephens 1970 claims this is more effective than the KS at detecting
    changes in the variance of a distribution; the KS is (he claims) more
    sensitive at detecting changes in the mean.
    If cdf was obtained from data by fitting, then fpp is not correct and
    it will be necessary to do Monte Carlo simulations to interpret D.
    D should normally be independent of the shape of CDF.
"""

#See ex1c.py for comments explaining these steps here:

N=int(1e5)
randomdata=BoxMuller(N,mean=0,stdv=1.0)

dex=0.1 
start,stop=1,5 
cnt=np.ones(int((stop-start)/dex)+1)
mykuiper=np.zeros((len(cnt),3))
scipykuiper=np.zeros((len(cnt),2))

for i in range(len(cnt)):
    cnt[i]+=round(i*dex,2)
    dataslice=randomdata[:int(10**cnt[i])]
    mykuiper[i]=kuiperstest(dataslice) # get D+,D-,pval from my Kuiper test
    scipykuiper[i,0]=kstest(dataslice,'norm',alternative='greater')[0] # get astropy D+
    scipykuiper[i,1]=kstest(dataslice,'norm',alternative='less')[0] # get scipy D-
   

#Make a plot comparing the D+and D- statistics 
plt.figure()
plt.subplot(2,1,1)
plt.title("Kuiper Test (D+ & D-)")
plt.plot(cnt,mykuiper[:,0],label="My Kuiper D+ statistic")
plt.plot(cnt,scipykuiper[:,0],label="Scipy Kuiper D+ statistic")
plt.ylabel("D+ statistic")
#plt.yscale("log")
#plt.ylim(top=(10**0.2))
plt.legend()

plt.subplot(2,1,2)
plt.plot(cnt,mykuiper[:,1],label="My Kuiper D- statistic")
plt.plot(cnt,scipykuiper[:,1],label="Scipy D- statistic")
plt.xlabel("$log_{10}(N_{points})$")
plt.ylabel("D- statistic")
#plt.yscale("log")
#plt.ylim(top=(10**0.2))
plt.legend()


plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig("./plots/1d_kuipertest_lohi.png")
plt.clf()



#make the same plot as 1(c):

dex=0.1 
start,stop=1,5 
cnt=np.ones(int((stop-start)/dex)+1)
mykuiper=np.zeros((len(cnt),2))
scipykuiper=mykuiper.copy()
from math import erf
from astropy.modeling.functional_models import Gaussian1D 
for i in range(len(cnt)):
    cnt[i]+=round(i*dex,2)
    dataslice=randomdata[:int(10**cnt[i])]
    kuipertemp=kuiperstest(dataslice) # get D+,D-,pval from my Kuiper test
    mykuiper[i]=kuipertemp[0]+kuipertemp[1],kuipertemp[2] # store those
    Dplus,p_greater=kstest(dataslice,'norm',alternative='greater') # get scipy D,Pval
    Dminus,p_less=kstest(dataslice,'norm',alternative='less') # for kuiper implementation
    print(p_greater-p_less)
    scipykuiper[i]=Dplus+Dminus,p_greater-p_less #reimplemented Kuiper via SciPy KS
    #astropykuiper[i]=kuiper(dataslice, GaussianCDF) # get astropy D,pval #failed attempt with astropy


plt.subplot(2,1,1)
plt.title("Kuiper Test: Box-Muller Normal rvs")
plt.plot(cnt,mykuiper[:,1],label="My Kuiper p-value")
plt.plot(cnt,scipykuiper[:,1],label="SciPy p-value")
plt.xlabel("$log_{10}(N_{points})$")
plt.ylabel("p-value")
plt.hlines(0.05,0.8,5.2,linewidth=0.8,linestyles=':')
plt.text(4.35,10**-1.6,'2-$\sigma$ rejection',color='r')
plt.hlines(0.003,0.8,5.2, linewidth=0.8,linestyles='--')
plt.text(4.35,10**-2.82,'3-$\sigma$ rejection',color='r')
plt.yscale("log")
plt.ylim((10**-3,10**0.2))
plt.legend(loc=1)

plt.subplot(2,1,2)
plt.plot(cnt,mykuiper[:,0],label="My Kuiper V statistic")
plt.plot(cnt,scipykuiper[:,0],label="SciPy V statistic")
plt.xlabel("$log_{10}(N_{points})$")
plt.ylabel("V statistic")
plt.yscale("log")
plt.ylim(top=(10**0.2))
plt.legend()


plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig("./plots/1d_kuipertest.png")
plt.clf()

