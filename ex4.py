import numpy as np

from ex4functions import romberg


def H(z,Om_m=0.3,Om_L=0.7,H_0=7.16e-11):
    return (H_0**2.*(Om_m*(1.+(1./z))**3.+Om_L))**0.5
def growthfactorintegrand(z,H):
    return (1/z**2.)*(1.+(1./z))/(H**3.)
      

Om_m=0.3
H_0=7.16e-11
Om_L=0.7

growthfactor=(5.*Om_m*H_0**2.*H(50)/2)*romberg(growthfactorintegrand,0,1/50.,order=5,acc=1e-5)

print(growthfactor)



#def f(x):
#    return x-np.log(x)
#print(romberg(f,5,50,order=10,acc=1e-5))

#romberg is accurate to 1/n^8 in the 3rd level of combinations

#the improper integral of the growth factor must be have a change of variables in redshift
# in order to make the integral converge. Do this by replacing z with 1/t, and multiplying
# the integrand by 1/t^2


#https://www.math.usm.edu/lambers/mat460/fall09/lecture33.pdf
    

