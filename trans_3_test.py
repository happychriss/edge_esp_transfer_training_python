import numpy as np
import requests
import sys, os, random
import tensorflow as tf
import datetime, os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools
import logging
import io
from tensorflow import keras
from sklearn import metrics
from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout, Conv1D, Conv2D, Flatten, Reshape, MaxPooling1D, MaxPooling2D, \
    BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorboard.plugins.hparams import api as hp
import os
import glob
import shutil
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout, Conv1D, Conv2D, Flatten, Reshape, MaxPooling1D, MaxPooling2D, \
    BatchNormalization
from tensorflow.keras.optimizers import Adam
import pandas as pd
import matplotlib
import constants as c

FEATURE_VALIDATION= 'wav_myvoice_validate'
SOURCE_H5_MODEL=os.path.join(c.MODELS_FOLDER,"ffc_16kb_shift_v1_32")
X_validate = np.load(os.path.join(c.FEATURES_FOLDER, 'x_' + FEATURE_VALIDATION) + '.npy')
Y_validate = np.load(os.path.join(c.FEATURES_FOLDER, 'y_' + FEATURE_VALIDATION) + '.npy')
Y_validate = tf.keras.utils.to_categorical(Y_validate, len(c.RESULT_CLASSES))
model = keras.models.load_model(SOURCE_H5_MODEL)
# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(X_validate, Y_validate, batch_size=128)
print("test loss, test acc:", results)

y_pred = model.predict(X_validate)

cm = metrics.confusion_matrix(c.RESULT_CLASSES, y_pred)

plt.imshow(cm, cmap=plt.cm.Blues)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.xticks([], [])
plt.yticks([], [])
plt.title('Confusion matrix ')
plt.colorbar()
plt.show()