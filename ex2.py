import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.colors import LogNorm


from ex1functions import BoxMuller
from ex2functions import createfourierplane,imagescale
from scipy.fftpack import ifft2


#choose an even grid size to put nyquist frequency in the center
N=1024 # gives sampling interval of 1/N in frequency space, and a Nyquist frequency 1/2*(1/N)

Nyquist_k=int(N/2)


#initialize grid of fourier plane with complex vectors of form Y=|Y|(cos(phi) + i*sin(phi))

# Store array of fourier coefficients in a conjugate symmetry to enforce that the inv fourier transform will be real: Y(k)=y_re + iy_im with Y(-k)=Y*(k) --> -k_x - ik_y = k_x - ik_y
# (Nyquist frequency and zero frequency have no negative frequency counterpart in discrete space, there's no "-0" components in kx or ky)
#fourier_coeff=np.concatenate((np.arange(0,1+Nyquist_k,1),np.arange(-1+Nyquist_k,0,-1)))
fourier_coeff=np.concatenate((np.arange(0,1+Nyquist_k,1),np.arange(1-Nyquist_k,0,1)))

#even_coeff=np.concatenate((np.arange(0,1+Nyquist_k,1),np.arange(1-Nyquist_k,0,1)))
#odd_coeff=np.roll(even_coeff,-int((len(even_coeff)/4)-1))
#fourier_coeff=[even_coeff,odd_coeff]
print(fourier_coeff)
# BoxMuller normal rvs with variance P(k,n)=sqrt(k_x^2+k_y^2)^n/2 and mean 0
# results in a Rayleight distributed fourier space (and Gaussian real space)


fourierplane1,kspace=createfourierplane(fourier_coeff,BoxMuller,N,n=-1.,returnk=True)
fourierplane2=createfourierplane(fourier_coeff,BoxMuller,N,n=-2.)
fourierplane3=createfourierplane(fourier_coeff,BoxMuller,N,n=-3.)


realplane1=ifft2(fourierplane1)
realplane2=ifft2(fourierplane2)
realplane3=ifft2(fourierplane3)


#plot the input space of fourier coefficients
plt.figure()
plt.subplot(2,2,1)
plt.imshow(np.real(kspace),origin='lower')
plt.title("$k_{real}$",fontsize=16)
plt.colorbar(fraction=0.046, pad=0.04)
plt.subplot(2,2,2)
plt.imshow(np.imag(kspace),origin='lower')
plt.title("$k_{imag}$",fontsize=16)
plt.colorbar(fraction=0.046, pad=0.04)
plt.subplot(2,2,3)
plt.imshow((np.real(kspace))**2.,origin='lower')
plt.title("abs($k_{real}$)",fontsize=16)
plt.colorbar(fraction=0.046, pad=0.04)
plt.subplot(2,2,4)
plt.imshow((np.imag(kspace))**2.,origin='lower')
plt.title("abs($k_{imag}$)",fontsize=16)
plt.colorbar(fraction=0.046, pad=0.04)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig("./plots/2_kspacecomponents.png")
plt.clf()

plt.imshow((np.real(kspace)+np.imag(kspace))**2.,origin='lower')
plt.title("abs(k)",fontsize=16)
plt.colorbar(fraction=0.046, pad=0.04)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig("./plots/2_kspace.png")
plt.clf()







#plot the resulting fourier plane and inverse FFT resulting in Gaussian random field
fourierplanes=[fourierplane1,fourierplane2,fourierplane3]
realplanes=[realplane1,realplane2,realplane3]

for i in range(3):
    plt.subplot(1,2,1)
    plt.ylabel("$P(k) \propto k^{"+str(-(i+1))+"}$",fontsize=14)
    plt.title("abs(Complex Fourier Plane)")
    plt.xlabel("1/Mpc")
    plt.imshow(abs(fourierplanes[i]),norm=LogNorm(),origin='lower')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(1,2,2)
    plt.title('abs(Gaussian Random Field)')
    plt.xlabel("Mpc")
    plt.imshow(abs(realplanes[i]),norm=LogNorm(),origin='lower')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig("./plots/2_fourier_and_realplane_n"+str(-(i+1))+".png")
    plt.clf()

    



#compare the real and imaginary parts of inverse FFT

plt.figure()
for i in range(3):
    plt.subplot(1,2,1)
    cmin,cmax=np.min(np.real(realplanes[i])),np.max(np.real(realplanes[i]))
    plt.title("real(Gaussian Random Field)")
    plt.ylabel("$P(k) \propto k^{"+str(-(i+1))+"}$",fontsize=14)
    plt.xlabel("Mpc")
    realpart=imagescale(realplanes[i],'real')
    plt.imshow(np.real(realplanes[i]),vmin=cmin,vmax=cmax,origin='lower')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(1,2,2)
    plt.title("imag(Gaussian Random Field)")
    plt.xlabel("phase")
    plt.imshow(np.imag(realplanes[i]),vmin=cmin,vmax=cmax,origin='lower') # scale to real part's colorbar
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig("./plots/2_realplane_imagrealpartse_n"+str(-(i+1))+".png")
    plt.clf()



#account for the zero mode frequency (an average over the whole gaussian field) and the Nyquist frequency (the smallest scale mode, which is half the sampling frequency aka half the distance between grid points)

#store fourier coefficients
