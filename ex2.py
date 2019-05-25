import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.colors import LogNorm


from ex1functions import BoxMuller
from ex2functions import createfourierplane,imagescale
from scipy.fftpack import ifft2


#choose an even grid size to put nyquist frequency in the center
N=10 # gives sampling interval of 1/N in frequency space, and a Nyquist frequency 1/2*(1/N)

Nyquist_k=int(N/2)


#initialize grid of fourier plane with complex vectors of form Y=|Y|(cos(phi) + i*sin(phi))

# Store array of fourier coefficients in an even symmetry to enforce symmetry that the fourier transform will be real: Y(-k)=Y*(k)
# (Nyquist frequency and zero frequency have no negative frequency counterpart in discrete space, there's no "-0" components in kx or ky)
#fourier_coeff=np.concatenate((np.arange(0,1+Nyquist_k,1),np.arange(-1+Nyquist_k,0,-1)))
fourier_coeff=np.concatenate((np.arange(0,1+Nyquist_k,1),np.arange(1-Nyquist_k,0,1)))


# BoxMuller normal rvs with variance P(k,n)=sqrt(k_x^2+k_y^2)^n/2 and mean 0
# results in a Rayleight distributed fourier space (and Gaussian real space)


fourierplane1,kxspace,kyspace=createfourierplane(fourier_coeff,BoxMuller,N,n=-1.,kspace=True)
fourierplane2=createfourierplane(fourier_coeff,BoxMuller,N,n=-2.)
fourierplane3=createfourierplane(fourier_coeff,BoxMuller,N,n=-3.)


realplane1=ifft2(fourierplane1)
realplane2=ifft2(fourierplane2)
realplane3=ifft2(fourierplane3)


#plot the input space of fourier coefficients
plt.subplot(1,2,1)
plt.imshow(kxspace,origin='lower')
plt.title("$k_x$",fontsize=16)
plt.subplot(1,2,2)
plt.imshow(kyspace,origin='lower')
plt.title("$k_y$",fontsize=16)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig("./plots/2_kspace.png")
plt.clf()




#plot the resulting fourier plane and inverse FFT resulting in Gaussian random field
fourierplanes=[fourierplane1,fourierplane2,fourierplane3]
realplanes=[realplane1,realplane2,realplane3]
dummypltcnt=0
for i in range(3):
    plt.subplot(3,2,i+1+dummypltcnt)
    plt.ylabel("$P(k) \propto k^{"+str(-(i+1))+"}$",fontsize=13)
    if i==0:
        plt.title("abs(Complex Fourier Plane)")
    if i==2:
        plt.xlabel("1/Mpc")
    plt.imshow(np.log10(abs(fourierplanes[i])),origin='lower')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(3,2,i+2+dummypltcnt)
    if i==0:
        plt.title('abs("Complex" Gaussian Field)')
    if i==2:
        plt.xlabel("Mpc")
    plt.imshow(np.log10(abs(realplanes[i])),origin='lower')
    plt.colorbar(fraction=0.046, pad=0.04)
    dummypltcnt+=1

    
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig("./plots/2_fourier_and_realplane.png")
plt.clf()


#compare the real and imaginary parts of inverse FFT

plt.figure()
dummypltcnt=0
for i in range(3):
    plt.subplot(3,2,i+1+dummypltcnt)
    cmin,cmax=np.min(np.real(realplanes[i])),np.max(np.real(realplanes[i]))
    if i==0:
        plt.title("real(Gaussian Random Field)")
    plt.ylabel("$P(k) \propto k^{"+str(-(i+1))+"}$",fontsize=13)
    if i==2:
        plt.xlabel("Mpc")
    realpart=imagescale(realplanes[i],'real')
    plt.imshow(np.real(realplanes[i]),vmin=cmin,vmax=cmax,origin='lower')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(3,2,i+2+dummypltcnt)
    if i==0:
        plt.title("imag(Gaussian Random Field)")
    if i==2:
        plt.xlabel("phase")
    plt.imshow(np.imag(realplanes[i]),vmin=cmin,vmax=cmax,origin='lower') # scale to real part's colorbar
    dummypltcnt+=1

    
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig("./plots/2_realplane_imagrealparts.png")
plt.clf()



#account for the zero mode frequency (an average over the whole gaussian field) and the Nyquist frequency (the smallest scale mode, which is half the sampling frequency aka half the distance between grid points)

#store fourier coefficients
