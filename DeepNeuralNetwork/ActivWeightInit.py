import numpy as np 
import matplotlib.pyplot as plt 
from keras import Sequential
from keras.layers import Dense, Activation
from keras.initializers import Zeros, RandomNormal
from keras.initializers import glorot_normal, glorot_uniform 
from keras.initializers import he_normal

n_input = 784
n_dense = 256

b_init = Zeros()

#w_init = RandomNormal(stddev = 1.0)
#w_init = glorot_normal()
w_init = he_normal()

model = Sequential()
model.add(Dense(n_dense, 
                input_dim=n_input,
                kernel_initializer=w_init,
                bias_initializer=b_init))
#model.add(Activation('sigmoid'))
#model.add(Activation('tanh'))
model.add(Activation('relu'))

x = np.random.random((1,n_input))
a = model.predict(x)

plt.figure("ReLU Activation  //  HE_Weight_Initialization")
plt.hist(np.transpose(a))
plt.show()