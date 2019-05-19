import numpy as np


#fourth order RK
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

def intode(func1,func2,integrator,start,stop,step,initcond=(3.,2.)):
    #make empty solution array(s)
    D=np.zeros(int((stop-start)/step)+1) # +1 for reaching 1000years
    t=D.copy()
    #set up init vals
    tcurr=start
    Dcurr,zcurr=initcond # (a.k.a. D(1) and D'(1))
    #integrate the coupled ode's
    for i in range(len(t)):
        t[i],D[i]=tcurr,Dcurr # store current approximation for spatial density
        Dnew,znew=RK4(func1,func2,tcurr,Dcurr,zcurr,step) # calculate solution at next step
        Dcurr,zcurr=Dnew,znew # update
        tcurr+=step # increment time
    return t,D # density solution over the time stop-start
