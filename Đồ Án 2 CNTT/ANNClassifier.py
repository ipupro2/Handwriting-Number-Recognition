from sklearn.neural_network import MLPClassifier

def Train(X_train, y_train):
    MLP = MLPClassifier()
    MLP.fit(X_train, y_train)
    return MLP

def Predict(MLP, data):
    return MLP.predict(data.reshape(1,-1))[0]

def Test(X_test, y_test, MLP):
    print(MLP.score(X_test, y_test))