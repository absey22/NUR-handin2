import numpy as np


def createfourierplane(coefficients,randgenerator,size,n=-1.,kspace=False):
    kxspace=np.empty((size,size));kyspace=np.empty((size,size))
    fourier=np.empty((size,size),dtype=complex)

    for xi in range(size):
        for yi in range(size):
            if xi==0 and yi==0:
                kx,ky=np.mean(coefficients),np.mean(coefficients) #  at zero frequency, DC component is the mean of frequencies
            else:
                kx,ky=coefficients[xi],coefficients[yi] # gather the two fourier coefficients
            
            kxspace[yi,xi]=kx;kyspace[yi,xi]=ky # for plotting
            a,b=randgenerator(2,mean=0,stdv=(kx**2.+ky**2.)**(n/2.)) # draw 2 normal variates of variance given by input power spectrum
            fourier[yi,xi]=complex(a,b) #append those compoenents
    if kspace:
        return fourier,kxspace,kyspace
    else:
        return fourier


def imagescale(image,part='real'):
    if part=='real':
        image=np.real(image)
    elif part=='imag':
        image=np.image(image)
    return (image-np.min(image))/np.ptp(image)

