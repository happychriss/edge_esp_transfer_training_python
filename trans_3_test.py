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

model = keras.models.load_model('path/to/location')

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)
