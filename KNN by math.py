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

def predict_KNN(X_train,y_train,X_test,img):
    u=0 
    for w in range(0,14):
        w*=2
        for e in range(0,14):
            e*=2
            for r in range(w,w+2):
                for t in range(e,e+2):
                    u += img[r][t]
            img[w][e]=u/4
            u=0
    
    Arr = []
    for i in range(14):
        ARR = []
        for j in range(14):
            ARR.append(img[i*2][j*2])
        Arr.append(ARR)
    
    C = np.array(Arr)
    
    C = C.reshape(-1)
    
    u=0
    for q in range(60000):
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
    for i in range(60000):
       A.append([])
       for j in range(14):
           A[i].append([])
           for k in range(14):
               A[i][j].insert(k,X_train[i][j*2][k*2])
                
    B=np.array(A)
    
    B=B.reshape(60000,-1)
    
    x=0
    for i in range(196):
        x += (B[0][i]-C[i])**2
    
    min = math.sqrt(x)
    
    x=0
    k=0
    for i in range(60000):
        for j in range(196):
            x += (B[i][j]-C[j])**2
        x = math.sqrt(x)
        if min > x:
            min = x
            k = i
        x=0
     
    return y_train[k] 


X_train, y_train = load_mnist('data/', kind='train')
X_test, y_test = load_mnist('data/', kind='t10k')


img = np.array(X_test[random.randrange(10000)])
print("INPUT IMAGE :")
plt.imshow(img, cmap='Greys', interpolation='nearest')
plt.show()


x = predict_KNN(X_train,y_train,X_test,img)
print('PREDICT NUMBER : %d '% (x))

            






    
