import numpy as np
import matplotlib.pyplot as plt



#to keep track of new indices of which split up DFT's correspond to original DFT's over k index, you can write the index k in binary format, reverse it in binary, and you get the index of the FFT coefficients.
def bitreversal(x):
    print("BIT REVERSAL")
    print(x.size%2)
    bitrevx=x.copy()
    if x.size==1:
        return x
    if x.size%2!=0: # ensure that the array is a power of two in length
        bitrevx=np.pad(bitrevx, (x.size%2,0), mode='constant',constant_values=0.0)
    else:        
        for k in range(x.size):
            print("---",k)
            bink=bin(k)[2:].zfill(4)
            print(bink)
            revk=int(bink[::-1],2)
            print(revk)
            bitrevx[revk]=x[k]
        print("reversed:",bitrevx)
        return bitrevx

#recursive Cooley-Tukey FFT in 1D
#Cooley Tukey algorithm rearranges the DFT of the function into two parts: a sum over the even-numbered indices and a sum over the odd-numbered indices
def fft(data):
    print("FFT")
    print("transform:",data)
    N=data.size
    #data=bitreversal(data) # bit reverse swap appropriate elements
    k=np.arange(0,N,1)
    if N>2:
        e=k[0:N:2] # slice for even elements of current array
        evencomponents=fft(data[e])
        o=k[1:N:2] # slice for odd elements
        oddcomponenets=fft(data[o])
    split1=np.zeros(int(N/2))
    split2=split1.copy()
    for i in range(split1.shape[0]): 
        W=np.exp(complex(0,-2.*np.pi*i/N))
        split1[i]=evencomponents[i]+W*oddcomponents[i]
        split2[i]=evencomponents[i]-W*oddcomponents[i]
        print(split1[i],split2[i])
    fourier=split1+split2 # the array containing the fourier transform of data
    return fourier

def function(x):
    return x**2.

print(fft(function(np.arange(1,57))))

#plt.plot(np.arange(1,50),function(np.arange(1,50)))
#plt.show()


