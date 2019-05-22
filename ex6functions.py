import numpy as np



def featurescale(x):
    return (x-np.mean(x))/np.std(x)


#logistic regression

def sigmoid(z):
    return 1./( 1.+np.exp(-1.*z) )


def hypothesis(features,parameters):
    bestestimate=features.dot(parameters)
    return sigmoid(bestestimate)
    #return bestestimate

def lossfunction(bestestimate,label):#label=dataset[:,2]
    #bestestimate is the hypothesis
    return -1.* ( label * np.log(bestestimate) + (1.-label)*np.log(1.-bestestimate) )

def costfunction(lossfunction):
    m=len(lossfunction) # equivalent to the # of samples trained on
    return (1./m)*np.sum(lossfunction)

def gradientdescent(parameters,hypothesis,label,samples,learningrate=0.1):
    m=len(samples) # number of samples trained on
    return parameters - (learningrate/m)*np.sum((hypothesis-label)[:,None]*samples,axis=0)
    #return parameters - (learningrate/m)*np.sum(np.matmul((hypothesis-label),samples),axis=0)


#testing
#count the successes and failure of classifier based on acutal labeling, and resulting F1 score
def f1score(datalabels,regoutput):
    truepos,trueneg=0,0
    falsepos,falseneg=0,0
    for i in range(len(regoutput)):
        if datalabels[i]==1. and regoutput[i]>0.5:
            truepos+=1
        if datalabels[i]==1. and regoutput[i]<0.5:
            falseneg+=1
        if datalabels[i]==0. and regoutput[i]<0.5:
            trueneg+=1
        if datalabels[i]==0. and regoutput[i]>0.5:
            falsepos+=1
    checksum=truepos+trueneg+falsepos+falseneg
    print("F1-score results:")
    print("TP:",truepos,", TN:",trueneg)
    print("FP:",falsepos,", FN:",falseneg)
    if checksum!=len(regoutput):
        print("checksum failed")
    f1=(2.*truepos)/(2.*truepos+trueneg+falsepos+falseneg)
    return f1,truepos,trueneg,falsepos,falseneg

#from solving equation of line: theta0*1 + theta1*x1 + theta2*x2 = 0.5 = yhat
def decisionboundary(x,b,w1,w2): #calculate bondary, is a line in case of 2 input neurons
    return -(w1 / w2) * x + ((0.5-b) / w2) # (just an equation of a line)
