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
plt.subplot(2,2,1,facecolor='gold')
plt.title(feature_names[3]+"  vs.  "+feature_names[2])
plt.plot(features[:,3][features[:,1]>=10.],features[:,2][features[:,1]>=10.],'ro')
plt.plot(features[:,3][features[:,1]<10.],features[:,2][features[:,1]<10.],'bo')
plt.subplot(2,2,2,facecolor='gold')
plt.title(feature_names[0]+"  vs.  "+feature_names[2])
plt.plot(features[:,0][features[:,1]>=10.],features[:,2][features[:,1]>=10.],'ro',label="Long GRBs")
plt.plot(features[:,0][features[:,1]<10.],features[:,2][features[:,1]<10.],'bo',label="Short GRBs")
plt.legend()
plt.subplot(2,2,3)
plt.title(feature_names[4]+"  vs.  "+feature_names[2])
plt.plot(features[:,4][features[:,1]>=10.],features[:,2][features[:,1]>=10.],'ro')
plt.plot(features[:,4][features[:,1]<10.],features[:,2][features[:,1]<10.],'bo')
plt.subplot(2,2,4)
plt.title(feature_names[4]+" vs. "+feature_names[6])
plt.plot(features[:,4][features[:,1]>=10.],features[:,6][features[:,1]>=10.],'ro')
plt.plot(features[:,4][features[:,1]<10.],features[:,6][features[:,1]<10.],'bo')

plt.tight_layout(pad=0.3, w_pad=0.3, h_pad=1.0)
plt.savefig("./plots/6_featurecomparison.png")
plt.clf()



#Exclude features from training set based on feature vs. feature behavior (and set missing features to zero):
# 0=Redshift    1=T90     2=log (M*/Msolar)    3=SFR    4=log (Z/Zsolar)   5=SSFR    6=AV
# most useful feature relations: (0,2) (2,3) (2,4) (4,6)
exclude_ind=[1,4,5,6]

#get the remaining desired features:
desired_features=np.setdiff1d(np.arange(0,n),exclude_ind)

if len(desired_features)!=3 or len(exclude_ind)!=4:
    print("Training not setup for more (or less) than 2 inputs.")
    exit()


#define desired training features
x1 = features[:,desired_features[0]]
x2 = features[:,desired_features[1]]
x3 = features[:,desired_features[2]]

#save a copy for plotting later
x1unscaled=x1.copy()
x2unscaled=x2.copy()
x3unscaled=x3.copy()

#clean and scale features
x1=featurescale(x1)
x2=featurescale(x2)
x3=featurescale(x3)

#report how amny feature samples were removed from each desired feature
print("Due to missing data samples were removed (from the data set of",m,"total samples):")
print("From",feature_names[desired_features[0]],":",len(x1[x1==0.]),"/",m,"samples removed.")
print("From",feature_names[desired_features[1]],":",len(x2[x2==0.]),"/",m,"samples removed.")
print("From",feature_names[desired_features[2]],":",len(x3[x3==0.]),"/",m,"samples removed.")



#OPTIONALLY reshape featues to polynomial, set this to change the combination method of features:
# 1: w0*1 + w1*x1 + w2*x2 + w3*x3
# 2: w0*1 + w1*x1 + w2*x2 + w3*x1**2 + w4*x2**2 + w5*x1*x2 + ...
polynomialorder=1

#redefine the features to asubset of the original data
if polynomialorder==1: # both augmented feature matrices for the bias parameter
    features=np.asarray([np.ones(len(x1)),x1,x2,x3]).T  # for basic linear combination
elif polynomialorder==2:
    features=np.asarray([np.ones(len(x2)),x2,x3,x2**2.,x3**2.,x2*x3]).T # order 2 feature combination



#Perform the logistic regression:

#initialize parameters to zero
parameters=np.zeros(features.shape[1])

#create initial best estimate yhat:
h_i=hypothesis(features,parameters)

#compute the initial cost function:
cost_i=costfunction(lossfunction(h_i,labels))


err_th=1e-6 # small threshold from T12.1
err=1e1 # initialize error to arbitrary large value
errlist=[err] # for terminating gradient descent
costfun=[cost_i] # for plotting


while err>err_th:
    #set learning rate
    alpha=0.01
    #update parameters via GD in logistic reg
    parameters=gradientdescent(parameters,h_i,labels,features,learningrate=alpha)
    #create new estimate yhat from those parameters:
    h_new=hypothesis(features,parameters)
    #compute the loss function from that estimation:
    loss_new=lossfunction(h_new,labels)
    #calculate the resulting cost:
    cost_new=costfunction(loss_new)
    #calculate error after this iteration
    err=abs(cost_new-cost_i)
    #store error, cost
    errlist.append(err)
    costfun.append(cost_new)
    #update cost,current hypothesis
    cost_i=cost_new
    h_i=h_new

#display the fit parameters via GD:
print("---------")
print("Using the features shape: w0*1 + w1*x1 + w2*x2 + w3*x3")
print("the parameters found via Gradient Descent in Logistic Regression are:")
weights=parameters[np.nonzero(parameters)]# grab the weights after the bias (will fail with more than 2 input weights in exlcude_ind
bias=weights[0]
print("Bias weight:",bias)
if polynomialorder==1:
    dummycnt=0
    for w in weights[1:]:
        print(feature_names[desired_features[dummycnt]],"weight:",w)
        dummycnt+=1
elif polynomialorder==2:
    for w in range(len(weights[1:])):
        print("Weight",w+1,":",weights[w])

        
#save the result of the regression performance
plt.suptitle("Log. Regression Performance: "+feature_names[desired_features[0]]+","+feature_names[desired_features[1]]+","+feature_names[desired_features[2]])
plt.subplot(2,1,1)
plt.plot(np.arange(len(costfun)),costfun,label='Cost (learning rate = '+str(alpha)+')')
plt.ylabel("J($\Theta$)")
plt.legend()
plt.subplot(2,1,2)
plt.plot(np.arange(len(errlist)),errlist,color='C1',label='Error')
plt.hlines(err_th,0,len(errlist)+10,color='k',linestyle=":",label="Desired Convergence = $10^{-6}$")
plt.xlabel("Iteration")
plt.ylabel("J($\Theta_{new}$) - J($\Theta_{i}$)")
plt.yscale("log")
plt.legend()

#plt.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
plt.savefig("./plots/6_classificationperformance.png")
plt.clf()

#calculate the output of the logistic regression which is yhat:
#(the resulting best hypothesis based on the GD parameters
#output=sigmoid( features.dot(parameters) )
output=hypothesis(features[:,1:],parameters[1:])


#test the classifier's accuracy
print("---------")
f1=f1score(labels,output)
print("Using",len(parameters),"weight(s) in logistic regression (including a bias) the resulting F1-score")
print("of this classifier is",round(100*f1[0],1),"%.")
print("It correctly classified",f1[1],"/",np.count_nonzero(labels==1.0)," of the known long GRBs and",f1[2],"/",np.count_nonzero(labels==0.0),"of the known short GRBs.")
print("(Training on a total of",m,"samples of GRBs.)")
#from sklearn.metrics import f1_score
#f1pkg=f1_score(labels,np.rint(output))
#print(f1pkg)


#the resulting decision boundaries
T90=dataset[:,1]

plt.title("Classifier 1 Results: "+feature_names[desired_features[0]]+" & "+feature_names[desired_features[1]]+" (and a bias)")
plt.plot(x1unscaled[T90>=10.],x2unscaled[T90>=10.],'ro',label="Long GRBs")
plt.plot(x1unscaled[T90<10.],x2unscaled[T90<10.],'bo',label="Short GRBs")
if polynomialorder==1:
    x1space=range(int(max(x1unscaled)))
    for threshold in np.arange(0.1,1.0,0.1):
        if threshold==0.5:
            plt.plot(x1space,decisionboundary(x1space,weights[0],weights[1],weights[2],threshold),linewidth=2.5,linestyle="--",label='$\hat{y}$ = 0.5')
        else:
            plt.plot(x1space,decisionboundary(x1space,weights[0],weights[1],weights[2],threshold),color='gray',linestyle=":")
plt.xlabel(feature_names[desired_features[1]])
plt.xlabel(feature_names[desired_features[0]])
plt.ylabel(feature_names[desired_features[1]])
plt.legend()

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig("./plots/6_decisionboundary_x1x2.png")
plt.clf()

plt.title("Classifier 2 Results: "+feature_names[desired_features[1]]+" & "+feature_names[desired_features[2]]+" (and a bias)")
plt.plot(x2unscaled[T90>=10.],x3unscaled[T90>=10.],'ro',label="Long GRBs")
plt.plot(x2unscaled[T90<10.],x3unscaled[T90<10.],'bo',label="Short GRBs")
if polynomialorder==1:
    x2space=range(int(max(x2unscaled)))
    for threshold in np.arange(0.1,1.0,0.1):
        if threshold==0.5:
            plt.plot(x2space,decisionboundary(x2space,weights[0],weights[2],weights[3],threshold),linewidth=2.5,linestyle="--",label='$\hat{y}$ = 0.5')
        else:
            plt.plot(x2space,decisionboundary(x2space,weights[0],weights[2],weights[3],threshold),color='gray',linestyle=":")
            plt.xlabel(feature_names[desired_features[1]])
plt.ylabel(feature_names[desired_features[2]])
plt.yscale("log")
plt.legend()

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig("./plots/6_decisionboundary_x2x3.png")
plt.clf()
