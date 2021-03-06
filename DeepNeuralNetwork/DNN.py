import keras 
from keras.datasets import mnist 
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD 
from keras.initializers import Zeros, RandomNormal
from keras.initializers import glorot_normal, glorot_uniform
from tensorflow.python.keras.callbacks import TensorBoard
from matplotlib import pyplot as plt

tensorboard = TensorBoard("logs/deep-net")

(X_train, y_train), (X_valid, y_valid) = mnist.load_data()

X_train = X_train.reshape(60000, 784).astype("float32") 
X_valid = X_valid.reshape(10000, 784).astype("float32")

X_train /= 255 
X_valid /= 255


n_classes = 10
y_train = keras.utils.to_categorical(y_train, n_classes)
y_valid = keras.utils.to_categorical(y_valid, n_classes)


model = Sequential()

model.add(Dense(128, activation='relu', input_shape=(784,)))
model.add(BatchNormalization())

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1, validation_data=(X_valid, y_valid), callbacks=[tensorboard])
model.evaluate(X_valid, y_valid)