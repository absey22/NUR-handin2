import numpy as np
import matplotlib.pyplot as plt


# ==========================  4   ==========================
from ex6functions import featurescale,sigmoid,f1score
from ex6functions import hypothesis,lossfunction,costfunction,gradientdescent


import pandas as pd
#df = pd.read_table("GRBs.txt",skiprows=2)#,usecols=(2,3,4,5,6,7,8,9),dtype=float)
df=pd.read_csv("GRBs.txt",delim_whitespace=True,skiprows=[1],na_values=[-1])
print(df)

exit()
#dataset =  np.array(df)

#Label the data via the T90 threshold:
# SHORT --> labeleddata[:,:,i] = 0 (T90<10s)
# LONG=  --> labeleddata[:,:,i] = 1 (T90>10s)
flags=np.where(dataset[:,2]>10.,1.0,0.0)


labeleddata=np.hstack((dataset,flags[:,None]))
print(labeleddata)
exit()
#this training set has 235 training examples
# or m=235 sample GRBs with variables of redshift,log(M),SFR,log(Z),SSFR,AV a.k.a. n=6 labels, or features
#dataset=np.loadtxt("GRBs.txt",skiprows=2,usecols=(2,3,4,5,6,7,8))

features=dataset[:,1:]
m=features.shape[0]
n=features[0].shape[0]



#feature scaling
for i in range(len(features[0,:])):
    features[:,i]=featurescale(features[:,i])



#reshape featues to polynomial
features=np.asarray([np.ones(features.shape[0]),features[:,0],features[:,1],features[:,0]**2.,features[:,1]**2.,features[:,0]*features[:,1]]).T
print(features[0])
#for basic linear combination
#features=np.hstack((np.ones(features.shape[0])[:,None],features))  # add a column in beginning to account for shape of having a bias weight (theta_0 or p[0], in parameters)


#initialize parameters
parameters=np.asarray([0.,0.,0.,0.,0.,0.])
#parameters=np.asarray([0.,0.,0.])





 #get labels of binary classification
labels=dataset[:,-1]


#create initial best estimate yhat:
h_i=hypothesis(features,parameters)
#compute the initial cost function:
cost_i=costfunction(lossfunction(h_i,labels))

err_th=1e-6
err=cost_i
errlist=[]
costfun=[]

#update parameters via GD
while err>err_th:
    #compute new parameters:
    parameters=gradientdescent(parameters,h_i,labels,features)
    print(parameters)
    #create new estimate yhat:
    h_new=hypothesis(features,parameters)
    #print(h_new[::int(len(h_new)/10)])
    #compute the loss function:
    loss_new=lossfunction(h_new,labels)
    #calculate the resulting cost:
    cost_new=costfunction(loss_new)
    
    err=abs(cost_i-cost_new)
    #err=cost_new
    errlist.append(err)
    costfun.append(cost_new)
    #update cost
    cost_i=cost_new
    #update the current hypothesis
    h_i=h_new
print(parameters)

plt.plot(np.arange(len(errlist)),errlist,label='err')
plt.plot(np.arange(len(costfun)),costfun,label='cost')
plt.legend()
plt.show()



output=sigmoid( features.dot(parameters) )

print(output.shape)


plt.plot(labels,output,'ro')
plt.show()



f1=f1score(labels,output)
print(f1)

