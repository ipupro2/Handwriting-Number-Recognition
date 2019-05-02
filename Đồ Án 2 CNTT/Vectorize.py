import numpy as np

def NormalVectorize(A):
    return A.reshape(len(A),-1)

def MeanVectorize14(X_train):
    u=0
    for q in range(0,len(X_train)):
        for w in range(0,14):
            w*=2
            for e in range(0,14):
                e*=2
                for r in range(w,w+2):
                    for t in range(e,e+2):
                        u += X_train[q][r][t]
                X_train[q][w][e]=u/4
                u=0
    for q in range(0,len(X_train)):
        for w in range(0,14):
            for e in range(0,14):
                X_train[q][w][e]=X_train[q][w*2][e*2]
        
    A=[]
    for i in range(0,len(X_train)):
        A.append([])
        for j in range(0,14):
            A[i].append([])
            for k in range(0,14):
                A[i][j].insert(k,X_train[i][j][k])
    return np.array(A)

def MeanVectorize7(X_train):
    u=0
    for q in range(0,len(X_train)):
        for w in range(0,7):
            w*=4
            for e in range(0,7):
                e*=4
                for r in range(w,w+4):
                    for t in range(e,e+4):
                        u += X_train[q][r][t]
                X_train[q][w][e]=u/16
                u=0
    for q in range(0,len(X_train)):
        for w in range(0,7):
            for e in range(0,7):
                X_train[q][w][e]=X_train[q][w*4][e*4]
    
    A=[]
    for i in range(0,len(X_train)):
        A.append([])
        for j in range(0,7):
            A[i].append([])
            for k in range(0,7):
                A[i][j].insert(k,X_train[i][j][k])
                
    return np.array(A)

def MaxVectorize14(X_train):
    for q in range(0,len(X_train)):
        for w in range(0,14):
            w*=2
            for e in range(0,14):
                e*=2
                max = X_train[q][w][e]
                for r in range(w,w+2):
                    for t in range(e,e+2):
                        if max < X_train[q][w][e]:
                            max = X_train[q][w][e]
                X_train[q][w][e]= max
             
    for q in range(0,len(X_train)):
        for w in range(0,14):
            for e in range(0,14):
                X_train[q][w][e]=X_train[q][w*2][e*2]
    
    A=[]
    for i in range(0,len(X_train)):
        A.append([])
        for j in range(0,14):
            A[i].append([])
            for k in range(0,14):
                A[i][j].insert(k,X_train[i][j][k])
                
    return np.array(A)

def MaxVectorize7(X_train):
    for q in range(0,len(X_train)):
        for w in range(0,7):
            w*=4
            for e in range(0,7):
                e*=4
                max = X_train[q][w][e]
                for r in range(w,w+4):
                    for t in range(e,e+4):
                        if max < X_train[q][r][t]:
                            max = X_train[q][r][t]
                X_train[q][w][e]= max
    for q in range(0,len(X_train)):
        for w in range(0,7):
            for e in range(0,7):
                X_train[q][w][e]=X_train[q][w*4][e*4]
    
    A=[]
    for i in range(0,len(X_train)):
        A.append([])
        for j in range(0,7):
            A[i].append([])
            for k in range(0,7):
                A[i][j].insert(k,X_train[i][j][k])
                
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
    
def MedianVectorize14(X_train):
    for q in range(0,len(X_train)):
        for w in range(0,14):
            w*=2
            for e in range(0,14):
                e*=2
                for r in range(w,w+2):
                    for t in range(e,e+2):
                        u = median(transarr(X_train[q][r][t]))
                X_train[q][w][e]=median(transarr(u))
    return X_train

def MedianVectorize7(X_train):
    for q in range(0,len(X_train)):
        for w in range(0,7):
            w*=4
            for e in range(0,7):
                e*=4
                for r in range(w,w+4):
                    for t in range(e,e+4):
                        u = median(transarr(X_train[q][r][t]))
                X_train[q][w][e]=median(transarr(u))
    
    for q in range(0,len(X_train)):
        for w in range(0,7):
            for e in range(0,7):
                X_train[q][w][e]=X_train[q][w*4][e*4]      
    
    A=[]
    for i in range(0,len(X_train)):
        A.append([])
        for j in range(0,7):
            A[i].append([])
            for k in range(0,7):
                A[i][j].insert(k,X_train[i][j][k])
                
    return np.array(A)