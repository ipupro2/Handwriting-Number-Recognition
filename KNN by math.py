import matplotlib.pyplot as plt
from PIL import Image 
import os
import numpy as np
import gzip
import math
import random 

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)
    
    with gzip.open(labels_path, 'rb') as lbpath:
        lbpath.read(8)
        buffer = lbpath.read()
        labels = np.frombuffer(buffer, dtype=np.uint8)
    with gzip.open(images_path, 'rb') as imgpath:
        imgpath.read(16)
        buffer = imgpath.read()
        images = np.frombuffer(buffer, dtype=np.uint8).reshape(len(labels), 28, 28).astype(np.float64)
    
    return images, labels

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
        
    A=[]
    for i in range(0,len(X_train)):
        A.append([])
        for j in range(0,14):
            A[i].append([])
            for k in range(0,14):
                A[i][j].insert(k,X_train[i][j*2][k*2])
    return np.array(A)

def NormalVectorize(A):
    return A.reshape(len(A),-1)


X_train, y_train = load_mnist('data/', kind='train')
X_test, y_test = load_mnist('data/', kind='t10k')

img = np.array(X_test)

X_train = MeanVectorize14(X_train)
X_test = MeanVectorize14(X_test)

X_train = NormalVectorize(X_train)
X_test = NormalVectorize(X_test)


def predict_KNN(X_train,y_train,X_test_y):
    x=0
    for i in range(196):
        x += (X_train[0][i]-X_test_y[i])**2
    
    min = math.sqrt(x)
    
    x=0
    k=0
    for i in range(60000):
        for j in range(196):
            x += (X_train[i][j]-X_test_y[j])**2
        x = math.sqrt(x)
        if min > x:
            min = x
            k = i
        x=0
     
    return y_train[k] 



## Xuat anh ra man hinh :
print("INPUT IMAGE :")
y = random.randrange(10000)
plt.imshow(img[y], cmap='Greys', interpolation='nearest')
plt.show()

##Xu ly chinh
print('LOADING..............')
x = predict_KNN(X_train,y_train,X_test[y])
print('PREDICT NUMBER : %d '% (x))

            






    
