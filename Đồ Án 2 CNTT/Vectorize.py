import numpy as np

def NormalVectorize(A):
    return A.reshape(len(A),-1)

def MeanVectorize(X_train, width):
    A = []
    u=0
    for q in range(0,len(X_train)):
        ARR = []
        for w in range(0,28,28//width):
            Arr = []
            for e in range(0,28,28//width):
                for r in range(w,w+28//width):
                    for t in range(e,e+28//width):
                        u += X_train[q][r][t]/((28/width)**2)
                Arr.append(u)
                u=0
            ARR.append(Arr)    
        A.append(ARR)  
    return np.array(A)

def MaxVectorize(X_train, width):
    A = []
    for q in range(0,len(X_train)):
        ARR = []
        for w in range(0,28,28//width):
            Arr = []
            for e in range(0,28,28//width):
                max = X_train[q][w][e]
                for r in range(w,w+28//width):
                    for t in range(e,e+28//width):
                        if max < X_train[q][r][t]:
                            max = X_train[q][r][t]
                Arr.append(max)
            ARR.append(Arr)    
        A.append(ARR)    
    return np.array(A)

def MedianVectorize(X_train, width):
    
    A = []
    for q in range(0,len(X_train)):
        ARR = []
        for w in range(0,28,28//width):
            Arr = []
            for e in range(0,28,28//width):
                u = []
                for r in range(w,w+28//width):
                    for t in range(e,e+28//width):
                        u.append(X_train[q][r][t])
                Arr.append(u)
            ARR.append(Arr)    
        A.append(ARR)  
    return np.array(A)

def median(lst): 
    """" Ham tinh median """
    sortedLst = sorted(lst) 
    lstLen = len(lst) 
    index = (lstLen - 1) // 2 
    if (lstLen % 2): 
     return sortedLst[index] 
    else: 
     return (sortedLst[index] + sortedLst[index + 1])/2.0 


def MinVectorize(X_train, width):
    A = []
    for q in range(0,len(X_train)):
        ARR = []
        for w in range(0,X_train.shape[1],28//width):
            Arr = []
            for e in range(0,X_train.shape[2],28//width):
                min = X_train[q][w][e]
                for r in range(w,w+28//width):
                    for t in range(e,e+28//width):
                        if min > X_train[q][r][t]:
                            min = X_train[q][r][t]
                Arr.append(min)
            ARR.append(Arr)    
        A.append(ARR)    
    return np.array(A)
