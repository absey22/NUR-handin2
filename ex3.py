import numpy as np
import matplotlib.pyplot as plt


# ==========================  3   ==========================
from ex3functions import intode,RK4


def f(t,D,z,Om_m=1.0): # (dz/dt=-4/3t * z + 2/3 Om_m/t^2 * D = f)
    return -(4./(3.*t))*z + (2.*Om_m/(3.*t**2.))*D

def g(t,D,z): # (z(t)=dD/dt = g)
    return z

#def f(t,D,z):
#    return z

#def g(t,D,z):
#    return 6.*D-z


#integrate from t=1yr to t=1000yrs
t0=1.; t_end=1000.
h=t0/(365)   #step size = 1 hour



case1=intode(f,g,RK4,start=t0,stop=t_end,step=h,initcond=(3.,2.))
case2=intode(f,g,RK4,start=t0,stop=t_end,step=h,initcond=(10.,-10.))
case3=intode(f,g,RK4,start=t0,stop=t_end,step=h,initcond=(5.,0.))


#x=np.linspace(t0,t_end,len(t))
#plt.plot(x,np.exp(-3.*x)+2.*np.exp(2.*x),label="analytical")
plt.plot(case1[0],case1[1],label='case1')
plt.plot(case2[0],case2[1],label='case2')
plt.plot(case3[0],case3[1],label='case3')
plt.xlabel("time (years)")
plt.ylabel("Density growth")
plt.yscale("log")
plt.xscale("log")
plt.ylim((10**-10,10**15))
plt.legend()
plt.show()
