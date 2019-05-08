import matplotlib.pyplot as plt
import os
import numpy as np
import gzip
import Vectorize as v
import AverageClassifier
import KNNClassifier
import ANNClassifier

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
X_train, y_train = load_mnist('data/', kind='train')
X_test, y_test = load_mnist('data/', kind='t10k')
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
X_train = v.MeanVectorize14(X_train)
X_test = v.MeanVectorize14(X_test)
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(10):
    img = X_train[y_train == i][0]
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
X_train = v.NormalVectorize(X_train)
X_test = v.NormalVectorize(X_test)

"""Average Classifier"""
AC = AverageClassifier.Train(X_train, y_train)
print("Average Classifier Accurary:", AverageClassifier.Test(X_test,y_test,AC)*100, "%")

"""KNN Classifier"""
print("KNN Predict Classifier:")
print("KNN")
KNNClassifier.Test(X_test, y_test, X_train, y_train)
print("KNN with Lib")
KNN = KNNClassifier.KNNbyLib(X_train, y_train)
KNNClassifier.KNNByLibTest(X_test, y_test, KNN)


"""ANN Classifier"""
ANN = ANNClassifier.Train(X_train, y_train)
print("ANN Classifier Accurary:")
ANNClassifier.Test(X_test, y_test, ANN)