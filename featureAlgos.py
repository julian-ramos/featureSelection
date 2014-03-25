#In this file will reside all of the feature
#selection algorithms

import numpy as np
import matplotlib.pylab as plt

def entropyVector(x):
    '''
    Computes the entropy of a vector of discrete values
    '''
    vals=np.bincount(x)
    #excluding the empty bins at the end and start of x
    vals=vals[1:-2]
    den=np.sum(vals)
    probs=vals/np.float(den)
    entro=-np.sum([i*np.log2(i) for i in probs if i!=0])/np.float(np.log2(np.size(vals,0)))
#     plt.plot(x)
#     plt.show()
    return entro
    


def informationGain(x,y):
    '''
    This implementation of information gain
    simply binerizes the data and then
    calculates information gain following the
    elements of information theory book equation in page
    x : features to be analyzed, where rows are data points and columns correspond to features
    y : class labels
    '''
    x=np.array(x)
    y=np.array(y)
    
    xMax=np.max(x,0)
    xMin=np.min(x,0)
    bins=[]
    
    #Discretized data
    xD=[]
    
    for i in range(len(xMin)):
        #-1e-5 and+1e-5 added so that the extremes are included in the 
        #bins, this however generates two empty bins that's why the last
        #[1:-2]
        bins.append(np.linspace(xMin[i]-1e-5,xMax[i]+1e-5, num=10))
        xD.append(np.digitize(x[:,i],bins[i]))
#         entros.append(entropy(xD[i]))
    
    # Now I need to calculate the join entropy for 
    # the class and the feature for that should be counted 
    # the times the different values of the class and
    # the feature as discretized occur. I could use here the
    # find multiple function or I could join the two 
    # vectors into one array and then just check them both??
    
    
#     plt.plot(x[:,0])
#     plt.plot(xD[0])
#     plt.show()
        
    print(entropyVector(xD[0]))
#     plt.hist(xD,10)
#     plt.hold(True)
    plt.hist(x*10,10)
    plt.show()
    


if __name__=='__main__':
    x=np.random.rand(900,3)
    y=np.round(np.random.rand(900,1))
    informationGain(x,y)
    
    
    