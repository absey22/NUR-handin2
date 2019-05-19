import numpy as np
import matplotlib.pyplot as plt




#def f(t,D,z,Om_m=1.0): # (dz/dt=-4/3t * z + 2/3 Om_m/t^2 * D = f)
#    return -(4./3*t)*z + (2.*Om_m/3.*t**2.)*D

#def g(t,D,z): # (z(t)=dD/dt = g)
#    return z

def f(t,D,z):
    return z

def g(t,D,z):
    return 6.*D-z

#===================
#fourth order RK

#integrate from t=1yr to t=1000yrs
t0=0.; t_end=10.
h=0.1     #step size

D=np.zeros(int((t_end-t0)/h)+1) # +1 for reaching 1000years
t=D.copy()

Dcurr=3. #initial conditions
zcurr=1.
tcurr=t0 #integrate from zero


#take two first order DE's (functions of t, D, and z) and increment via RK4
def RK4(fode1,fode2,t0,d0,z0,step):
    k1=step*fode1(t0,d0,z0)
    l1=step*fode2(t0,d0,z0)
    k2=step*fode1(t0+0.5*step,d0+0.5*k1,z0+0.5*l1)
    l2=step*fode2(t0+0.5*step,d0+0.5*k1,z0+0.5*l1)
    k3=step*fode1(t0+0.5*step,d0+0.5*k2,z0+0.5*l2)
    l3=step*fode2(t0+0.5*step,d0+0.5*k2,z0+0.5*l2)
    k4=step*fode1(t0+step,d0+k3,z0+l3)
    l4=step*fode2(t0+step,d0+k3,z0+l3)
    #step in D and z
    dnext=d0+(1./6.)*(k1 + 2.*k2 + 2.*k3 + k4)
    znext=z0+(1./6.)*(l1 + 2.*l2 + 2.*l3 + l4)
    return dnext,znext

for i in range(len(t)):
    t[i],D[i]=tcurr,Dcurr # store current approximation for spatial density
    print(t[i],D[i])
    Dnew,znew=RK4(f,g,tcurr,Dcurr,zcurr,step=h)
    Dcurr,zcurr=Dnew,znew # update
    tcurr+=h

x=np.linspace(t0,t_end,len(t))
plt.plot(x,np.exp(-3.*x)+2.*np.exp(2.*x),label="analytical")
plt.plot(t,D,":",label='RK4')
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.show()