import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from matplotlib import pyplot as plt

(X_train, y_train), (X_valid, y_valid) = mnist.load_data()

X_train.shape

plt.figure(figsize=(5,5))
for k in range(600):
    plt.subplot(10, 60, k+1)
    plt.imshow(X_train[k], cmap='Greys')
    plt.axis('off')
plt.tight_layout()
plt.show()