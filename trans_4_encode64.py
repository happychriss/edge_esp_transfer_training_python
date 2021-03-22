# opens the model and generate a h5 file b64 encoded to paste into edgeimpulse
# result is in rd_str

import base64
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout, Conv1D, Conv2D, Flatten, Reshape, MaxPooling1D, MaxPooling2D, BatchNormalization
from tensorflow import keras
import constants as c
import os


SOURCE_H5_MODEL=os.path.join(c.MODELS_FOLDER,"transfer_trained_v1_weights.h5")

classes_values = c.RESULT_CLASSES
classes = len(classes_values)

# Transfer binary into string
rd= open(SOURCE_H5_MODEL,'rb').read()
rd_64 = base64.b64encode(rd)
h5_model_str=rd_64.decode("UTF-8")
print(h5_model_str)

model_weights_str=h5_model_str

# from utf-8 back to binary
model_weights_b64=model_weights_str.encode('utf-8')
model_weights=base64.b64decode(model_weights_b64)
f = open("./models/model.h5", "wb")
f. write(model_weights)
f.close()

#model = keras.models.load_model('model.h5')
#model.summary()


# # model architecture
input_length = 828
model = Sequential()
model.add(Reshape((int(input_length / 18), 18), input_shape=(input_length, )))
model.add(Conv1D(13, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
model.add(Dropout(0.5))
model.add(Conv1D(23, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(classes, activation='softmax', name='y_pred'))
model.load_weights('./models/model.h5')
model.summary()


