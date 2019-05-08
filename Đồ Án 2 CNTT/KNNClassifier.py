import math
import random
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def Predict(X_train,y_train,data):
    x=0
    for i in range(196):
        x += (X_train[0][i]-data[i])**2
    
    min = math.sqrt(x)
    
    x=0
    k=0
    for i in range(60000):
        for j in range(196):
            x += (X_train[i][j]-data[j])**2
        x = math.sqrt(x)
        if min > x:
            min = x
            k = i
        x=0
     
    return y_train[k] 

def Test(X_test, y_test, X_train, y_train):
    for i in range(0, 10):
        img = X_test[random.randint(0,len(X_test))]
        print(Predict(X_train, y_train, img))
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