import numpy as np

def Train(X_train, y_train):
    SumVector = np.zeros((10,X_train.shape[1]))
    Count = [0,0,0,0,0,0,0,0,0,0]
    for i in range(0,len(X_train)):
        SumVector[y_train[i]] += X_train[i]
        Count[y_train[i]] += 1
    for i in range(0, 10):
        SumVector[i] /= Count[i]
    print(type(SumVector))
    return SumVector
    
def Predict(data, SumVector):
    minIndex = 0
    min = np.linalg.norm(data - SumVector[0])
    for i in range(1,10):
        cur = np.linalg.norm(data - SumVector[i])
        if cur < min:
            min = cur
            minIndex = i
    return minIndex

def Test(X_test, y_test, SumVector):
    count = 0
    for i in range(0,X_test.shape[0]):
        if Predict(X_test[i], SumVector) == y_test[i]:
            count += 1
    return count/X_test.shape[0]