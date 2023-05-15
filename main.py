import yfinance as yf
import numpy as np
import tensorflow as tf
from tensorflow import keras

apple = yf.Ticker("AAPL")
hist = apple.history(period="1d", start="2021-05-15", end="2023-05-15")
open_prices = hist["Open"]
data = np.array(open_prices)

n = 4

max_value = max(data)
min_value = min(data)

data = [(value - min_value) / (max_value - min_value) for value in data]
r = int(0.7*len(data))

x = []
y = []
x_test = []
y_test = []

while len(data) > n:
    str = data[:n]
    if r > 0:
        x.append(str)
        y.append(data[n])
    else:
        x_test.append(str)
        y_test.append(data[n])

    r = r - 1
    data = data[1:]

triv = y_test.copy()
triv.insert(0, y[-1])
triv.pop()

model = keras.models.Sequential([
    keras.layers.Dense(n, input_shape=(n,)),
    keras.layers.Dense(n, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

train_count = 1000

min_loss = float("inf")
best_model = None

for i in range(train_count):
    history = model.fit(x, y, validation_data=(x_test, y_test), epochs=100, verbose=0)
    val_loss = history.history['val_loss'][-1]

    if val_loss < min_loss:
        min_loss = val_loss
        best_model = model.get_weights()
        print ("На шаге", i+1, "мин. ошибка уменьшилась до", min_loss)
    else:
        print("На шаге", i+1, "мин. ошибка осталась", min_loss)
        model.reset_states()

model.set_weights(best_model)

mse = keras.metrics.mean_squared_error(y_test, triv)
print(f"Ошибка тривиальной модели: {mse}")

final = np.array(data).reshape(-1, n)
pred = model.predict(final, verbose=0).item() * (max_value - min_value) + min_value
print ("Предсказание нейросети:", pred)