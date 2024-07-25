import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#Sigmoid fonksiyonu tanimlandi
def sigmoid_numpy(X):
    return 1 / (1 + np.exp(-X))

#Loss
def log_loss(y_true, y_predicted):
    epsilon = 1e-15
    y_predicted_new = [max(i, epsilon) for i in y_predicted]
    y_predicted_new = [min(i, 1 - epsilon) for i in y_predicted_new]
    y_predicted_new = np.array(y_predicted_new)
    return -np.mean(y_true * np.log(y_predicted_new) + (1 - y_true) * np.log(1 - y_predicted_new))

#Class
class myNN:
    def __init__(self):
        self.w1 = 1
        self.w2 = 1
        self.bias = 0

#Fit fonksiyonu
    def fit(self, X, y, epochs, loss_threshold):
        self.w1, self.w2, self.bias = self.gradient_descent(X['age'], X['affordibility'], y, epochs, loss_threshold)

#Predict burda yapildi
    def predict(self, X_test):

        weighted_sum = self.w1 * X_test['age'] + self.w2 * X_test['affordibility'] + self.bias
        return sigmoid_numpy(weighted_sum)

    def gradient_descent(self, age, affordability, y_true, epochs, loss_threshold):
        w1 = w2 = 1
        bias = 0
        rate = 0.5
        n = len(age)
        for i in range(epochs):
            weighted_sum = w1 * age + w2 * affordability + bias
            y_predicted = sigmoid_numpy(weighted_sum)
            loss = log_loss(y_true, y_predicted)

            w1d = (1 / n) * np.dot(np.transpose(age), (y_predicted - y_true))
            w2d = (1 / n) * np.dot(np.transpose(affordability), (y_predicted - y_true))

            bias_d = np.mean(y_predicted - y_true)
            w1 = w1 - rate * w1d
            w2 = w2 - rate * w2d
            bias = bias - rate * bias_d

            if i % 50 == 0:
                print(f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')


            if loss <= loss_threshold:
                print(f'Epoch: {i}, w1: {w1}, w2:{w2}, bias: {bias}, loss:{loss}')

                break

        return w1, w2, bias


data = pd.read_csv('insurance_data.csv')
print(data.head())

X_train, X_test, y_train, y_test = train_test_split(data[['age', 'affordibility']], data.bought_insurance,
                                                    test_size=0.2, random_state=25)

X_train_scaled = X_train.copy()
X_train_scaled['age'] = X_train_scaled['age'] / 100

X_test_scaled = X_test.copy()
X_test_scaled['age'] = X_test_scaled['age'] / 100

model = keras.Sequential([
    keras.layers.Dense(1, input_shape=(2,), activation='sigmoid', kernel_initializer='ones', bias_initializer='zeros')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
evaluated = model.evaluate(X_test_scaled, y_test)
#model.fit(X_train_scaled, y_train, epochs=5000)




customModel = myNN()
customModel.fit(X_train_scaled, y_train, epochs=5000, loss_threshold=0.4631)

print("predict: ", customModel.predict(X_test_scaled))
print("x test sclade: ", model.predict(X_test_scaled)[:5])

print("evaluated: ",evaluated)
