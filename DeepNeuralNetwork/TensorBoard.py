import numpy as np
import tensorflow as tf
from tensorflow import keras
from time import time
from tensorflow.python.keras.callbacks import TensorBoard

tensorboard = TensorBoard(log_dir="logs/deep-net".format(time()))

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_image, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_image = test_image / 255.0

model = keras.Sequential([
        keras.layers.Flatten(input_shape = (28,28)),
        keras.layers.Dense(128, activation = tf.nn.relu),
        keras.layers.Dense(10, activation = tf.nn.softmax)
])

model.compile(optimizer = tf.train.AdamOptimizer(),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(train_images, train_labels, epochs = 5, callbacks = [tensorboard])

test_loss, test_acc = model.evaluate(test_image, test_labels)

