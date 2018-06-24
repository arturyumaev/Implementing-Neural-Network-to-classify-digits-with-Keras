import sklearn
import numpy as np
import scipy as sp
import keras

from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

def get_mnist_data():
    digits = load_digits()
    X = digits.data
    X = X / np.max(X)
    Y = digits.target
    norm_targets = Y
    y = np.zeros((len(Y), 10))

    for i in range(len(y)):
        y[i][Y[i]] = 1
    
    X_train, X_test, y_train, y_test, nt_train, nt_test = train_test_split(X, y, norm_targets, test_size=0.2)

    return X_train, X_test, y_train, y_test, nt_train, nt_test

X_train, X_test, y_train, y_test, nt_train, nt_test = get_mnist_data()

model = Sequential()
model.add(Dense(64, input_shape=(64, ), activation="elu"))
model.add(Dense(64, input_shape=(64, ), activation="elu"))
model.add(Dense(10, input_shape=(64, ), activation="softmax", ))

model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])

training_history = model.fit(X_train, y_train, epochs=100)
history_losses = training_history.history["loss"]
history_acc = training_history.history["acc"]

print("Validation accuracy:", np.mean(model.predict_classes(X_test) == nt_test))

plt.grid(True)
plt.plot(range(len(history_losses)), history_losses)
plt.plot(range(len(history_losses)), history_acc)
plt.legend(["categorical_crossentropy", "accuracy"])
plt.show()

