import numpy as np

from ex4functions import romberg,midpointrule


def Hubble(z,Om_m=0.3,Om_L=0.7,H_0=7.16e-11):
    return ( H_0**2. * (Om_m*(1.+z)**3.+Om_L) )**0.5

def GFintegrand_cov(z,Om_m=0.3,Om_L=0.7,H_0=7.16e-11):
    t=z**-1.# change of variables t = 1/z for improper integral
    H=Hubble(t)
    return (1/t**2.) * ( (1.+t)/ H**3. )

def GF(z1,z2,m,Om_m=0.3,Om_L=0.7,H_0=7.16e-11):
    awayfromsingularity=(5.*Om_m*H_0**2.*Hubble(z1)/2.)*romberg(GFintegrand_cov,(1./z2)+m,1./z1,order=10,acc=1e-5) # evaluate via romberg small distance m away from singularity (at 0, after c.o.v.)
    atsingularity=midpointrule(GFintegrand_cov,1./z2,m,n=3) # evaluate via ext. midpt from singularity to small distance m away
    totalintegral=atsingularity+awayfromsingularity
    return totalintegral

#romberg is accurate to 1/n^8 in the 3rd level of combinations

#the improper integral of the growth factor must be have a change of variables in redshift
# in order to make the integral converge. Do this by replacing z with 1/t, and multiplying
# the integrand by 1/t^2
#split the integral at m due to singularity at 0 after change of variables to get rid of improper integral
print(GF(50.,np.inf,m=1e-5))

#https://www.math.usm.edu/lambers/mat460/fall09/lecture33.pdf
    

