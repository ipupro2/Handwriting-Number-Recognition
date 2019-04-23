import matplotlib.pyplot as plt
import os
import numpy as np
import gzip
from sklearn import svm
from skimage.feature import hog
from skimage import io
import time
import random

from PIL import Image, ImageFilter

def imageprepare(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    # newImage.save("sample.png

    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    return tva

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

fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,) 
ax = ax.flatten()
for i in range(10):
    img = X_train[y_train == i][0]
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
time.sleep(0.2)
#X_train = X_train.reshape(len(y_train), -1)
#X_test = X_test.reshape(len(y_test), -1)
hogImageTrain = []
hogImageTest = []
for image in X_train:
    hogImageTrain.append(hog(image, pixels_per_cell=(7,7), cells_per_block=(1,1), block_norm="L2"))
for image in X_test:
    hogImageTest.append(hog(image, pixels_per_cell=(7,7), cells_per_block=(1,1), block_norm="L2"))
hogImageTrain = np.asarray(hogImageTrain)
hogImageTest = np.asarray(hogImageTest)
print(hogImageTrain[0])
print("Training...")
time.sleep(0.2)
now = time.time()
clf = svm.LinearSVC(C = 10)
#clf = svm.SVC(gamma = 0.001)

clf.fit(hogImageTrain, y_train)
#clf.fit(X_train, y_train)
print("Mất %.2f giây để train"%(time.time() - now))
print("Bạn có muốn test?(Y/N)")
if input()=="Y":
    print("Testing...")
    time.sleep(0.2)
    now = time.time()
    score = clf.score(X_test, y_test) 
    print("Tỉ lệ chính xác %.2f" % score)
    print("Mất %.2f giây để test"%(time.time() - now))
else:
    print("No Testing")
print("Bạn có muốn mở file từ bên ngoài")
ans = input()
if ans!="N":
    while True:
        image = imageprepare(ans)
        image = np.asarray(image)
        image = image.reshape(28,28)
        plt.imshow(image, cmap='Greys', interpolation='nearest')
        image = hog(image, pixels_per_cell=(7,7), cells_per_block=(1,1), block_norm="L2")
        result = clf.predict(image.reshape(1, -1))[0]
        print("Số này là %.d" % result)
        ans = input()
#randomIndex = random.randint(-1,len(y_test)-1)
#result = int(clf.predict(hogImageTest[randomIndex].reshape(1,-1))[0])
#print("Số này là %.d" % result)
#plt.imshow(hogImageTest[randomIndex], cmap='Greys', interpolation='nearest')