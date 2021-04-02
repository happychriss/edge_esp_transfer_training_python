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

# MODE_BASELINE, tansfer, teach the transfer base model for transfer learning, SAVED_MODEL is generated
# MODE_MY, teach my-voice with transfer model a

###################################################################################################
# Input for training
FEATURES_TRAIN = 'wav_mytrain'
FEATURES_TEST = 'wav_mytest'

LOAD_WEIGHTS = 1  # 1 = load weights from model # 0 = dont load
LOAD_WEIGHTS_MODEL_NAME = "ffc_16kb_shift_v1_32_weights.h5"

SAVE_MODEL = 0  # 1 = save model, 0 = dont save
SAVED_MODEL_NAME = "ffc_16kb_shift_v1_32"

MODEL_LEARNING_RATE = 0.0005 #default is 0.001 for ffc, 0.0005 for transfer
MODEL_EPOCHS = 70 # 130 for ffc, 70 for transfer
###################################################################################################


# Load the TensorBoard notebook extension
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set random seeds for repeatable results
#RANDOM_SEED = 3
#random.seed(RANDOM_SEED)
#np.random.seed(RANDOM_SEED)
#tf.random.set_seed(RANDOM_SEED)

# Prepare Logging

shutil.rmtree(c.LOG_PATH)

logdir = os.path.join(c.LOG_PATH, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
file_writer_cm = tf.summary.create_file_writer(logdir)

classes_values = c.RESULT_CLASSES
classes = len(classes_values)

# Prepare Training and Testdata
X_train = np.load(os.path.join(c.FEATURES_FOLDER, 'x_' + FEATURES_TRAIN) + '.npy')
Y_train = np.load(os.path.join(c.FEATURES_FOLDER, 'y_' + FEATURES_TRAIN) + '.npy')
Y_train = tf.keras.utils.to_categorical(Y_train, classes)

X_test = np.load(os.path.join(c.FEATURES_FOLDER, 'x_' + FEATURES_TEST) + '.npy')
Y_test = np.load(os.path.join(c.FEATURES_FOLDER, 'y_' + FEATURES_TEST) + '.npy')
Y_test = tf.keras.utils.to_categorical(Y_test, classes)


# plt.hist(Y, bins=11)
# plt.show()
# Y = tf.keras.utils.to_categorical(Y - 1, classes)
# X_Edge, X_, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

input_length = X_train[0].shape[0]

g_train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
g_validation_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))


def train_test_model():
    def set_batch_size(batch_size, l_train_dataset, l_validation_dataset):
        my_train_dataset = l_train_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
        my_validation_dataset = l_validation_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
        return my_train_dataset, my_validation_dataset

    def plot_to_image(figure):
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(figure)
        buf.seek(0)

        digit = tf.image.decode_png(buf.getvalue(), channels=4)
        digit = tf.expand_dims(digit, 0)

        return digit

    def plot_confusion_matrix(cm, class_names):
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Accent)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
        threshold = cm.max() / 2.

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        return figure

    def log_confusion_matrix(epoch, logs):
        predictions = model.predict(X_test)
        predictions = np.argmax(predictions, axis=1)

        cm = metrics.confusion_matrix(np.argmax(Y_test, axis=1), predictions)
        figure = plot_confusion_matrix(cm, class_names=classes_values)
        cm_image = plot_to_image(figure)

        with file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)

    # model architecture
    model = Sequential()
    channels = 1
    columns = 18
    rows = int(input_length / (columns * channels))


    # model.add(Reshape((rows, columns, channels), input_shape=(input_length,)))
    # model.add(Conv2D(8, kernel_size=3, activation='relu', kernel_constraint=tf.keras.constraints.MaxNorm(1), padding='same'))
    # model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    # model.add(Dropout(0.5))
    # model.add(Conv2D(16, kernel_size=3, activation='relu', kernel_constraint=tf.keras.constraints.MaxNorm(1), padding='same'))
    # model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    # model.add(Dropout(0.5))
    # model.add(Flatten())
    # model.add(Dense(classes, activation='softmax', name='y_pred'))

    # base model, full trained with ff command set

    model.add(Reshape((int(input_length / 18), 18), input_shape=(input_length,)))

    if SAVE_MODEL==1:

        model.add(Conv1D(13, kernel_size=3, activation='relu', padding='same'))
        model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
        model.add(Dropout(0.25))
        model.add(Conv1D(23, kernel_size=3, activation='relu', padding='same'))
        model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(classes, activation='softmax', name='y_pred'))

    # target model, to be trained with my voice - loading the weights from base model
    if LOAD_WEIGHTS==1:

        model.add(Conv1D(13, kernel_size=3, activation='relu', padding='same',trainable = False))
        model.add(MaxPooling1D(pool_size=2, strides=2, padding='same',trainable = True))
        model.add(Dropout(0.25))
        model.add(Conv1D(23, kernel_size=3, activation='relu', padding='same',trainable = False))
        model.add(MaxPooling1D(pool_size=2, strides=2, padding='same',trainable = False))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(classes, activation='softmax', name='y_pred'))
        # load the weights
        model.load_weights(os.path.join(c.MODELS_FOLDER, LOAD_WEIGHTS_MODEL_NAME))

    # this controls the learning rate
    opt = Adam(lr=MODEL_LEARNING_RATE, beta_1=0.9, beta_2=0.999)

    # Some tuning, sa.mapper not available
    #    g_train_dataset = g_train_dataset.map(sa.mapper(), tf.data.experimental.AUTOTUNE)

    # this controls the batch size, or you can manipulate the tf.data.Dataset objects yourself
    BATCH_SIZE = 32
    train_dataset, validation_dataset = set_batch_size(BATCH_SIZE, g_train_dataset, g_validation_dataset)

    # train the neural network
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()

    metrics_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir,
                                                      histogram_freq=1,
                                                      write_graph=True,
                                                      write_images=True,
                                                      update_freq='epoch',
                                                      profile_batch=2,
                                                      embeddings_freq=1)

    confusion_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)
    callbacks = [metrics_callback, confusion_callback]
    #    callbacks=[]

    model.fit(train_dataset, epochs=MODEL_EPOCHS, validation_data=validation_dataset, verbose=2, callbacks=callbacks)

    if SAVE_MODEL==1:
        tmp_name = os.path.join(c.MODELS_FOLDER, SAVED_MODEL_NAME)
        model.save(tmp_name)
        model.save_weights(tmp_name + '_weights.h5')
        model.save(tmp_name + '.h5')
    return


print(logdir)
print("TensorFlow version: ", tf.__version__)
train_test_model()
