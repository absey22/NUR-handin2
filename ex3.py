import numpy as np
import matplotlib.pyplot as plt


# ==========================  3   ==========================
from ex3functions import intode,RK4


def f(t,D,z,Om_m=1.0): # (dz/dt=-4/3t * z + 2/3 Om_m/t^2 * D = f)
    return -(4./(3.*t))*z + (2.*Om_m/(3.*t**2.))*D

def g(t,D,z): # (z(t)=dD/dt = g)
    return z

#integrate from t=1yr to t=1000yrs
t0=1.; t_end=1000.
h=t0/365   #step size = 1 day


#integrate for 3 different initial value conditions, the above couple 1st ODE's
# using RK4 method.
case1=intode(g,f,RK4,start=t0,stop=t_end,step=h,initcond=(3.,2.))
case2=intode(g,f,RK4,start=t0,stop=t_end,step=h,initcond=(10.,-10.))
case3=intode(g,f,RK4,start=t0,stop=t_end,step=h,initcond=(5.,0.))

#using analytical method. (wolfram)
#analytical general solution: D(t)=at^2/3+b/t
# a,b found via plugging in initial values.
tspace=np.linspace(t0,t_end,int((t_end-t0)/h))
Anl_case1=3*tspace**(2./3.)
Anl_case2=10./tspace
Anl_case3=3*tspace**(2./3.)+2./tspace


plt.title("Linear Structure Growth")
#NUMERICAL
plt.plot(case1[0],case1[1],linewidth=3,label='Case 1')
plt.plot(case2[0],case2[1],linewidth=3,label='Case 2')
plt.plot(case3[0],case3[1],linewidth=3,label='Case 3')
#ANALYTICAL
plt.plot(tspace,Anl_case1,"--",label='Anl Case 1')
plt.plot(tspace,Anl_case2,"--",label='Anl Case 1')
plt.plot(tspace,Anl_case3,"--",label='Anl Case 1')

plt.xlabel("time (years)")
plt.ylabel("D(t)")
plt.xscale("log")
plt.yscale("log")
#plt.arrow(10**2,10**2,0,10**10,width=3,head_length=0.2*10**12,shape='full')
#plt.arrow(10**2,10**-1,0,-.99998*10**-1,width=3,head_length=0.000001999,shape='full')
#plt.text(2*10**2,10**9,"Collapse",fontsize=13,rotation=90)
#plt.text(2*10**2,10**-2,"Expansion",fontsize=13,rotation=90)
#plt.ylim((10**-10,10**15))
plt.legend(loc=3)


plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig("./plots/3_spatialdensitygrowth.png")
plt.clf()

plt.title("Growth at t < 10 years")
plt.plot(case1[0],case1[1],label='Case 1')
plt.plot(case2[0],case2[1],label='Case 2')
plt.plot(case3[0],case3[1],label='Case 3')
plt.xlabel("time (years)")
plt.ylabel("D(t)")
plt.xscale("log")
plt.yscale("log")
plt.xlim((8*10**-1,10**1))
plt.ylim((5*10**-1,5*10**1))
plt.legend()

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()
plt.savefig("./plots/3_spatialdensitygrowth_neart0.png")
plt.clf()

