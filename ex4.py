import numpy as np

def growthfactor(z,H,Om_m=0.3,H_0=7.16e-11):
    integrand=(1.+z)/(H**3.)
    return (5.*Om_m*H_0**2.*H/2)*integrand
    
def H(z,Om_m=0.3,Om_L=0.7):
    return H_0**2.*(Om_m*(1.+z)**3.+Om_L)
