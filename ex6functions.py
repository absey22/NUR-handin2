import numpy as np



def featurescale(x):
        return (x-np.mean(x))/np.std(x)


#logistic regression

def sigmoid(z):
    return 1./( 1.+np.exp(-1.*z) )


def hypothesis(data,parameters):
    p=parameters
    bestestimate=p[0]*1. + p[1]*data[:,0] + p[2]*data[:,1] \
                          + p[3]*(data[:,0])**2. +  p[4]*(data[:,1])**2. \
                           + p[5]*(data[:,0]*data[:,1])
    return sigmoid(bestestimate)
    #return bestestimate

def lossfunction(bestestimate,label):#label=dataset[:,2]
    #bestestimate is the hypothesis
    return -1.* ( label * np.log(bestestimate) + (1.-label)*np.log(1.-bestestimate) )

def costfunction(lossfunction):
    return (1./m)*np.sum(lossfunction)

def gradientdescent(parameters,hypothesis,label,samples,learningrate=0.1):
    return parameters - (learningrate/m)*np.sum((hypothesis-label)[:,None]*samples,axis=0)
    #return parameters - (learningrate/m)*np.sum(np.matmul((hypothesis-label),samples),axis=0)


#testing

def f1score(datalabels,regoutput):
    truepos,trueneg=0,0
    falsepos,falseneg=0,0
    for i in range(len(output)):
        if datalabels[i]==1. and regoutput[i]>0.5:
            truepos+=1
        if datalabels[i]==1. and regoutput[i]<0.5:
            falseneg+=1
        if datalabels[i]==0. and regoutput[i]<0.5:
            trueneg+=1
        if datalabels[i]==0. and regoutput[i]>0.5:
            falsepos+=1
    checksum=truepos+trueneg+falsepos+falseneg
    print("TP",truepos,"TN",trueneg)
    print("FP",falsepos,"FN",falseneg)
    if checksum!=len(regoutput):
        print("checksum failed")
    f1=(2.*truepos)/(2.*truepos+trueneg+falsepos+falseneg)
    return f1
