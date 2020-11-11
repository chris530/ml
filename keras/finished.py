#!/usr/local/bin/python3

import numpy as np

from tensorflow.keras.models import load_model
model = load_model('MyTrainedModel.h5')

model.summary()
#print(model.get_weights())


# should be 0
sample = np.array([[5,116,74,0,0,25.6,0.201,30]])
print(model.predict_classes(sample))

# should be 1
sample = np.array([[10,168,74,0,0,38,0.537,34]])
print(model.predict_classes(sample))
