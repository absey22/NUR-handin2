import numpy as np
import matplotlib.pyplot as plt


# ==========================  6   ==========================
from ex6functions import featurescale,sigmoid,f1score,decisionboundary
from ex6functions import hypothesis,lossfunction,costfunction,gradientdescent


dataset=np.genfromtxt("GRBs.txt",usecols=(2,3,4,5,6,7,8))


#get labels of binary classification
#Label the data via the T90 threshold:
# SHORT --> labeleddata[:,0] = 0. (T90<10s)
# LONG=  --> labeleddata[:,0] = 1. (T90>=10s)
labels=np.where(dataset[:,1]>=10.,1.0,0.0)

#append labels to last column of data --> labeled training set
dataset=np.hstack((dataset,labels[:,None]))

#this training set has 235 training examples
# or m=235 sample GRBs with variables of redshift,log(M/Msolar),SFR,log(Z/Zsolar),SSFR,AV a.k.a. n=7 labels, or features
features=dataset[:,:-1].copy()
m=features.shape[0]; n=features.shape[1]




feature_names=['Redshift','$T_{90}$','log($M*/M_{\odot}$)',"SFR",'log($Z/Z_{\odot}$)',"SSFR","AV"]

#examine features by plotting versus each other to see clustering with short/long behavior
for i in range(1,n):
    plt.subplot(2,n/2,i)
    if i==2:
         plt.subplot(2,n/2,i,facecolor='gold')
    plt.plot(features[:,0][features[:,1]>=10.],features[:,i][features[:,1]>=10.],'ro',label="Long GRBs")
    plt.plot(features[:,0][features[:,1]<10.],features[:,i][features[:,1]<10.],'bo',label="Short GRBs")
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[i])
    if i==1:
        plt.legend()
plt.tight_layout(pad=0.3, w_pad=0.3, h_pad=1.0)
plt.savefig("./plots/6_featurecomparison.png")
plt.clf()


#Exclude features from training set (and set missing features to zero):

# 0=Redshift    1=T90     2=log (M*/Msolar)    3=SFR    4=log (Z/Zsolar)   5=SSFR    6=AV
exclude_ind=[1,3,4,5,6] # set undesire features to zero so that they dont effect training
for ind in exclude_ind:
    features[:,ind]=0.0
    
#get the remaining desired features:
desired_features=np.setdiff1d(np.arange(0,n),exclude_ind)

if len(desired_features)!=2 or len(exclude_ind)!=5:
    print("Training not setup for more (or less) than 2 inputs.")
    exit()

#apply feature scaling to each feature
unscaledfeatures=features.copy()
for i in range(n):
    if i in exclude_ind: # dont bother calling the feature scaling on the excluded features
        continue
    features[:,i]=featurescale(features[:,i])


#reshape featues to polynomial
x1=features[:,desired_features[0]]
x2=features[:,desired_features[1]]


#set this to change the combination method of features:
# 1: w0*1 + w1*x1 + w2*x2
# 2: w0*1 + w1*x1 + w2*x2 + w3*x1**2 + w4*x2**2 + w5*x1*x2
polynomialorder=1


if polynomialorder==1: # both augmented feature matrices for the bias parameter
    features=np.asarray([np.ones(m),x1,x2]).T  # for basic linear combination
elif polynomialorder==2:
    features=np.asarray([np.ones(m),x1,x2,x1**2.,x2**2.,x1*x2]).T # order 2 feature combination





#LOGISTIC REGRESSION:

#initialize parameters to zero
parameters=np.zeros(features.shape[1]) # +1 for bias!! 

#create initial best estimate yhat:
h_i=hypothesis(features,parameters)
#compute the initial cost function:
cost_i=costfunction(lossfunction(h_i,labels))

err_th=1e-6 # small threshold from T12.1
err=1e4 # initialize error to arbitrary large value
errlist=[err] # for terminating gradient descent
costfun=[cost_i] # for plotting

#update parameters via GD in logistic reg
while err>err_th:
    #compute new parameters:
    parameters=gradientdescent(parameters,h_i,labels,features)
    #create new estimate yhat from those parameters:
    h_new=hypothesis(features,parameters)
    #compute the loss function from that estimation:
    loss_new=lossfunction(h_new,labels)
    #calculate the resulting cost:
    cost_new=costfunction(loss_new)
    #calculate error in this iteratoin
    err=abs(cost_new-cost_i)
    #store error, cost
    errlist.append(err)
    costfun.append(cost_new)
    #update cost,current hypothesis
    cost_i=cost_new
    h_i=h_new

#show the results:
print("Including the bias the parameters found via Gradient Descent in Logistic Regression are:")
bias=parameters[0]
if polynomialorder==1:
    wa,wb=parameters[1:][np.nonzero(parameters[1:])] # grab the weights after the bias (will fail with more than 2 input weights in exlcude_ind
    print("Bias:",bias)
    print("Feature",desired_features[0],"weight:",wa)
    print("Feature",desired_features[1],"weight:",wb)
elif polynomialorder==2:
    weights=parameters[1:][np.nonzero(parameters[1:])] # grab the weights after the bias (will fail with more than 2 input weights in exlcude_ind
    print("Bias 0 :",bias)
    for w in range(len(weights)):
        print("Weight",w+1,":",weights[w])

plt.suptitle("Logistic Regression Performance: ("+feature_names[desired_features[0]]+" & "+feature_names[desired_features[1]]+")")
plt.subplot(2,1,1)
plt.plot(np.arange(len(costfun)),costfun,label='Cost')
plt.ylabel("J($\Theta$)")
plt.legend()
plt.subplot(2,1,2)
plt.plot(np.arange(len(errlist)),errlist,color='C1',label='Error')
plt.hlines(err_th,0,len(errlist),color='k',linestyle=":",label="Desired Convergence")
plt.xlabel("Iteration")
plt.ylabel("J($\Theta_{new}$) - J($\Theta_{i}$)")
plt.yscale("log")
plt.legend()

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig("./plots/6_classificationperformance.png")
plt.clf()

#calculate the output of the logistic regression which is yhat:
output=sigmoid( features.dot(parameters) )
print(output)
#test the classifier's accuracy
print("---------")
f1=f1score(labels,output)
print("Using",len(parameters),"weight(s) in logistic regression (including a bias) the resulting F1-score")
print("of this classifier is",round(100*f1[0],1),"%. It correctly classified as long GRBs",f1[1],"of")
print("the known",np.count_nonzero(labels==1.0),"long GRBs (from a total of",m,"samples of GRBs.)")


#the resulting decision boundary

T90=dataset[:,1]
feature1=unscaledfeatures[:,desired_features[0]]; 
feature2=unscaledfeatures[:,desired_features[1]]
feature1space=np.linspace(min(feature1)-0.2,max(feature1)+0.2,100)  # for plotting the decision boundary

plt.title("Classifier Results: "+feature_names[desired_features[0]]+" & "+feature_names[desired_features[1]]+" (and a bias)")
plt.plot(feature1[T90>=10.],feature2[T90>=10.],'ro',label="Long GRBs")
plt.plot(feature1[T90<10.],feature2[T90<10.],'bo',label="Short GRBs")
if polynomialorder==1:
    plt.plot(feature1space,decisionboundary(feature1space,bias,wa,wb),"--",label='Decision Boundary')
plt.xlabel(feature_names[desired_features[0]])
plt.ylabel(feature_names[desired_features[1]])
plt.legend()

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig("./plots/6_decisionboundary.png")
