import numpy as np
import matplotlib.pyplot as plt

from ex1functions import I_0
print("Initial state set by seed in ex1functions.py:",I_0)


# ==========================  1(a)   ==========================
from ex1functions import Urng

#make a random realization of 1e6 uniformly drawn scalars
uniformdraw=Urng(size=int(1e6))

#(FORMATTING PARTIALLY BORROWED FROM MY EXERCISE 1)
#plot the resulting 1000 floats generated:
plt.figure()
plt.subplot(2,2,1)
plt.plot(uniformdraw[0:1000:2],uniformdraw[1:1001:2],'ro')
plt.xlabel("$x_i$")
plt.ylabel("$x_{i+1}$")
plt.title("$n_{draws}$ = 1,000")

#plot index vs. magnitude
plt.subplot(2,2,2)
plt.xlabel("index i")
plt.ylabel("$x_i$")
plt.title("$n_{draws}$ = 1,000")
plt.bar(np.arange(len(uniformdraw[0:1000])),uniformdraw[0:1000])

#histogram the 1,000,000 random numbers
plt.subplot(2,1,2)
freq,bins,_=plt.hist(uniformdraw,bins=20,edgecolor='black', linewidth=1.2)
binmidpts=0.5*(bins[1:] + bins[:-1])                  #find mid point of each bin
#plt.errorbar(binmidpts, freq, yerr=(freq)**0.5, fmt='none',label='Poissonian Std. dev.')
plt.hlines(50000,-0.02,1.02,linestyles=':',label='Ideal Uniform')
plt.hlines(50000+50000**0.5,-0.02,1.02,linestyles=':',color='r',label='+/-1-$\sigma$ (Poisson)')
plt.hlines(50000-50000**0.5,-0.02,1.02,linestyles=':',color='r')
plt.title("MWC & XOR Shift Uniform Variate ($n_{draws}$ = 1,000,000)")
plt.xlabel("True values (0.05 wide bins)")
plt.ylabel("Frequency")
plt.ylim(48000,51000)
plt.legend(loc=8)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig("./plots/1a_uniformityanalysis.png")
plt.clf()
