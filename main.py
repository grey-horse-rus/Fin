import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import metrics

n = 5
data = []

with open('data.txt', 'r') as f:
    for line in f:
        numbers = line.split()
        row = [float(num) for num in numbers]
        data.append(row)

data = np.array(data)
min_val = data.min()
max_val = data.max()
data_norm = (data - min_val) / (max_val - min_val)
l = len(data_norm[0])

r = int(len(data_norm) * 0.7)
x = []
y = []
x_test = []
y_test = []
triv = []
final = []
i = 0

while i < len(data_norm) - n:
    new_row = []
    if i < r:
        for j in range(n):
            row = data_norm[i+j]
            new_row.extend(row)
        x.append(new_row)
        y.append(data_norm[i+n])
    else:
        for j in range(n):
            row = data_norm[i+j]
            new_row.extend(row)
        x_test.append(new_row)
        y_test.append(data_norm[i + n])
        triv.append(data_norm[i + n - 1])
    i = i + 1

while i < len(data_norm):
    final.extend(data_norm[i])
    i = i + 1

x = np.array(x)
y = np.array(y)
x_test = np.array(x_test)
y_test = np.array(y_test)
final = np.array(final)

mse = metrics.mean_squared_error(y_test, triv).numpy().mean()

model = keras.models.Sequential([
    keras.layers.Dense(n*l, input_shape=(n*l,)),
    keras.layers.Dense(n*l),
    keras.layers.Dense(l)
])

model.compile(optimizer='adam', loss='mean_squared_error')

train_count = 10

min_loss = float("inf")
best_model = None

for i in range(train_count):
    history = model.fit(x, y, validation_data=(x_test, y_test), epochs=100, verbose=0)
    val_loss = history.history['val_loss'][-1]

    if val_loss < min_loss:
        min_loss = val_loss
        best_model = model.get_weights()
        print ("На шаге", i+1, "точность улучшилась до", min_loss / mse)
    else:
        print("На шаге", i+1, "точность по-прежнему", min_loss / mse)
        model.reset_states()

model.set_weights(best_model)

final = final.reshape(1, n*l)
pred = model.predict(final, verbose=0) * (max_val - min_val) + min_val
print (pred)