import numpy as np

#(BORROWED FROM EXERCISE 1)
def trapezoidrule(func,x1,x2,panels):
    h=(x2-x1)/panels
    I=0.5*(func(x1)+func(x2))
    for i in range(1,panels):
        I+=func(x1+i*h)
    return h*I

def midpointrule(func,a,b,n): # for n number of steps between a and b
    print("Midpoint from a =",a,"to b =",b)
    sections=1
    for j in range(n-1):
        sections*=3 # for adding more interior points
    d=(b-a)/(3.*sections)
    xevalpt=a+0.5*d
    I=0.0
    for i in range(sections-1):
        I+=func(xevalpt)
        xevalpt+=d # eval point spacing
    return d*I


def neville(I_estimate,order,acc): #benefits from Richardson Extrapolation of smaller and smaller error term with each order
    I=I_estimate # set initial trapezoid estimations
    for order in range(1,order): # loop over orders for approximation combinations
        for i in range(len(I)-order): # apply algorithm in place from I[i] and I[i+1] previous best two estimates
            #print(order,i,I[i],I[i+1])
            I[i]=(4.**(float(order))*I[i+1] - I[i]) / (4.**(float(order))-1.)
        err=abs(I[0]-I[1])
        if err<=acc:
            print("(Combination of trapezoid estimates converged to",acc,"in order",order,".)")
            return I[0]
    return I[0]

def romberg(func,a=1.,b=2.,order=3,acc=1e-5): 
    I_initial=np.zeros(order)
    print("Romberg from a =",a,"to b =",b)
    n=1 # initialize amount of sections
    for i in range(order): # generate order initial estimates via trapezoid rule
        I_initial[i]=trapezoidrule(func,a,b,n)
        n*=2 # half the step size each time (by doubling amount of sections)
    I_best=neville(I_initial,order,acc) # feed estimates to nevilles polyinterp
    return I_best




