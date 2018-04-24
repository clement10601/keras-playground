import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
import matplotlib.pyplot as plt # 可视化模块

tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

# create some data
X = np.linspace(-1, 1, 4500)
np.random.shuffle(X)    # randomize the data
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (4500, ))
# plot data
#plt.scatter(X, Y)
#plt.show()

X_train, Y_train = X[:3500], Y[:3500]
X_val, Y_val = X[3500:4000], Y[3500:4000]
X_test, Y_test = X[4000:], Y[4000:]

model = Sequential()
model.add(Dense(32, input_shape=(1,)))
model.add(Dense(1, input_shape=(32,)))
# choose loss function and optimizing method
model.compile(loss='mse', optimizer='sgd')

model.fit(x=X_train, y=Y_train, batch_size=32, epochs=100, callbacks=[tbCallBack])
# training
# print('Training -----------')
# for step in range(301):
    
#     cost = model.train_on_batch(X_train, Y_train)
#     if step % 100 == 0:
#         print('train cost: ', cost)

# plotting the prediction
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()
