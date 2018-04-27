import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras import optimizers
import matplotlib.pyplot as plt # 可视化模块

tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

# create some data
X = np.linspace(-1, 1, 600)
np.random.shuffle(X)    # randomize the data
Y = np.power(X, 3) - np.power(X, 2) + 0.5 * X + 2 + np.random.normal(0, 0.05, (600, ))
# plot data
plt.scatter(X, Y)
#plt.show()

X_train, Y_train = X[:350], Y[:350]
X_val, Y_val = X[350:400], Y[350:400]
X_test, Y_test = X[400:], Y[400:]

actfunc = 'selu'

model = Sequential()
model.add(Dense(16, input_dim=1, activation=actfunc))
#model.add(BatchNormalization())
model.add(Dense(16, activation=actfunc))
model.add(Dense(8, activation=actfunc))
model.add(Dense(4, activation=actfunc))
model.add(Dense(2 , activation=actfunc))
model.add(Dense(1 , activation=actfunc))
# model.add(Dense(1, input_shape=(32,)))
# choose loss function and optimizing method
#sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
model.compile(loss='mse', optimizer='sgd')

model.fit(x=X_train, y=Y_train, batch_size=45, epochs=100, callbacks=[tbCallBack])
# training
# print('Training -----------')
# for step in range(301):
    
#     cost = model.train_on_batch(X_train, Y_train)
#     if step % 100 == 0:
#         print('train cost: ', cost)

# plotting the prediction
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.scatter(X_test, Y_pred)
plt.show()
