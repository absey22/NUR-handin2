import numpy as np


def createfourierplane(coefficients,randgenerator,size,n=-1.,returnk=False):
    fourier=np.empty((size,size),dtype=complex)
    kspace=fourier.copy()
    coefficients_R,coefficients_I=coefficients,coefficients
    for xi in range(size):
        for yi in range(size):
            if xi==0 and yi==0:
                k=complex(np.mean(coefficients_R),np.mean(coefficients_I)) #  at zero frequency, DC component is the mean of frequencies
            else:
                k=complex(coefficients_R[xi],coefficients_I[yi]) # gather the two fourier coefficients
            
            kspace[yi,xi]=k # for plotting
            a,b=randgenerator(2,mean=0,stdv=(k.real**2.+k.imag**2.)**(n/2.))
            if xi==int(size/2) or yi==int(size/2):
                fourier[yi,xi]=complex(a,0.0) # set the Nyquist frequency to real
            else:
                fourier[yi,xi]=complex(a,b)
    if returnk:
        return fourier,kspace
    else:
        return fourier


def imagescale(image,part='real'):
    if part=='real':
        image=np.real(image)
    elif part=='imag':
        image=np.image(image)
    return (image-np.min(image))/np.ptp(image)

