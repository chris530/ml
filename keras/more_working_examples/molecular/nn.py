#!/usr/local/bin/python3.9

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
df = pd.read_csv('molecular_activity.csv')
properties = list(df.columns.values)
properties.remove('Activity')
X = df[properties]
y = df['Activity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(4,)),
    keras.layers.Dense(16, activation=tf.nn.relu),
	keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid),
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=1)

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

# Final

a= np.array([[4.02,70.86,62.05,7.0],[2.99,60.30,57.46,6.06]])
print(model.predict(a))
