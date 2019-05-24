import numpy as np



from ex1functions import BoxMuller

normaldraw=BoxMuller(N=10,mean=0.0,stdv=3.0)


print(normaldraw)

fourierplane=np.zeros()


#choose a grid size

#account for the zero mode frequency (an average over the whole gaussian field) and the Nyquist frequency (the smallest scale mode, which is half the sampling frequency aka half the distance between grid points)

#store fourier coefficients
