import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

x = np.array
([[2,3],
             [4,5],
             [1,2],
             [6,7]])

y = np.array([0,1,0,1])

model = Sequential()
model.add(Dense(4, input_dim=2, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(x, y, epochs=10)

import matplotlib.pyplot as plt
plt.plot(history.history["loss"])
plt.plot(history.history["accuracy"])
plt.show()