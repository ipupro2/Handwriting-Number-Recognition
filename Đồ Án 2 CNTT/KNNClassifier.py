import math
import random
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

def Predict(X_train,y_train,data,k):
    x=0
    A = []
    for i in range(len(X_train)):
        Arr = []
        for j in range(196):
            x += (X_train[i][j]-data[j])**2
        x = math.sqrt(x)
        Arr.append(x)
        Arr.append(y_train[i])
        A.append(Arr)
        x=0  
        
    A=np.array(A)
    A=A[A[:, 0].argsort()]
    
    ARR=np.zeros(10)
    count = 0
    for i in range(10):
        for j in range(k):
            if A[j][1] == i:
                count+=1
        ARR[i]=count
        count = 0
    return ARR.argmax()

def Test(X_test, y_test, X_train, y_train):
    for i in range(0, 10):
        img = X_test[random.randint(0,len(X_test))]
        print(Predict(X_train, y_train, img, 100))
        plt.imshow(img.reshape(14,14), cmap='Greys', interpolation='nearest')
        plt.show()
#        img = X_test[i]
#        if(Predict(X_train, y_train, img)==y_test[i]):
#            count+=1
#    print(count,"/",len(X_test))
#    print(time.time()-now, "s")

def KNNbyLib(X_train, y_train):
    clf = KNeighborsClassifier(n_neighbors = 3, weights = 'distance', algorithm = 'brute', p = 2)
    clf.fit(X_train, y_train)
    return clf

def KNNByLibTest(X_test, y_test, clf):
    for i in range(0, 10):
        img = X_test[random.randint(0,len(X_test))]
        print(clf.predict(img.reshape(1,-1)))
        plt.imshow(img.reshape(14,14), cmap='Greys', interpolation='nearest')
        plt.show()
