import math
import random
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

def distance(X_train,data):
    x=0;
    for i in range(data.shape[0]):
        x+=math.pow(X_train[i]-data[i],2)
    return math.sqrt(x)  

def findmax(A,k):
    A=np.array(A)
    A=A[A[:, 0].argsort()]
    ARR = []
    for i in range (k):
        ARR.append(int(A[i][1]))      
    ARR=np.array(ARR) 
    return np.bincount(ARR).argmax()
 

def Predict(X_train,y_train,data,k):

    A = []
    for i in range(len(X_train)):
        Arr = []
        Arr.append(distance(X_train[i],data))
        Arr.append(y_train[i])
        A.append(Arr)
    return findmax(A,k)

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
