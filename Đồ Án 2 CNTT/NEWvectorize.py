import numpy as np

def NormalVectorize(A):
    return A.reshape(len(A),-1)

def MeanVectorize14(X_train):
    A = []
    u=0
    for q in range(0,len(X_train)):
        ARR = []
        for w in range(0,28,2):
            Arr = []
            for e in range(0,28,2):
                for r in range(w,w+2):
                    for t in range(e,e+2):
                        u += X_train[q][r][t]/4
                Arr.append(u)
                u=0
            ARR.append(Arr)    
        A.append(ARR)  
    return np.array(A)

def MeanVectorize7(X_train):
    A = []
    u=0
    for q in range(0,len(X_train)):
        ARR = []
        for w in range(0,28,4):
            Arr = []
            for e in range(0,28,4):
                for r in range(w,w+4):
                    for t in range(e,e+4):
                        u += X_train[q][r][t]/16
                Arr.append(u)
                u=0
            ARR.append(Arr)    
        A.append(ARR)  
    return np.array(A)

def MaxVectorize14(X_train):
    A = []
    for q in range(0,len(X_train)):
        ARR = []
        for w in range(0,28,2):
            Arr = []
            for e in range(0,28,2):
                max = X_train[q][w][e]
                for r in range(w,w+2):
                    for t in range(e,e+2):
                        if max < X_train[q][r][t]:
                            max = X_train[q][r][t]
                Arr.append(max)
            ARR.append(Arr)    
        A.append(ARR)    
    return np.array(A)

def MaxVectorize7(X_train):
    A = []
    for q in range(0,len(X_train)):
        ARR = []
        for w in range(0,28,4):
            Arr = []
            for e in range(0,28,4):
                max = X_train[q][w][e]
                for r in range(w,w+4):
                    for t in range(e,e+4):
                        if max < X_train[q][r][t]:
                            max = X_train[q][r][t]
                Arr.append(max)
            ARR.append(Arr)    
        A.append(ARR)    
                
    return np.array(A)

def MedianVectorize14(X_train):
    
    A = []
    for q in range(0,len(X_train)):
        ARR = []
        for w in range(0,28,2):
            Arr = []
            for e in range(0,28,2):
                for r in range(w,w+2):
                    for t in range(e,e+2):
                        u = median(transarr(X_train[q][r][t]))
                Arr.append(u)
            ARR.append(Arr)    
        A.append(ARR)  
    return np.array(A)   

def MedianVectorize7(X_train):   
    A = []
    for q in range(0,len(X_train)):
        ARR = []
        for w in range(0,28,4):
            Arr = []
            for e in range(0,28,4):
                for r in range(w,w+4):
                    for t in range(e,e+4):
                        u = median(transarr(X_train[q][r][t]))
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
 
def transarr(arr):
    """" Ham vecto hoa """
    return arr.reshape(-1)

def MinVectorize7(X_train):
    A = []
    for q in range(0,len(X_train)):
        ARR = []
        for w in range(0,X_train.shape[1],4):
            Arr = []
            for e in range(0,X_train.shape[2],4):
                min = X_train[q][w][e]
                for r in range(w,w+4):
                    for t in range(e,e+4):
                        if min > X_train[q][r][t]:
                            min = X_train[q][r][t]
                Arr.append(min)
            ARR.append(Arr)    
        A.append(ARR)    
    return np.array(A)

def MinVectorize14(X_train):
    A = []
    for q in range(0,len(X_train)):
        ARR = []
        for w in range(0,X_train.shape[1],2):
            Arr = []
            for e in range(0,X_train.shape[2],2):
                min = X_train[q][w][e]
                for r in range(w,w+2):
                    for t in range(e,e+2):
                        if min > X_train[q][r][t]:
                            min = X_train[q][r][t]
                Arr.append(min)
            ARR.append(Arr)    
        A.append(ARR)    
    return np.array(A)

    
