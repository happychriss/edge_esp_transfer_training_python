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

# Load the TensorBoard notebook extension
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set random seeds for repeatable results
RANDOM_SEED = 3
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Prepare Logging
LOG_PATH = "./logs"
shutil.rmtree(LOG_PATH)

logdir = os.path.join(LOG_PATH, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
file_writer_cm = tf.summary.create_file_writer(logdir)

# Prepare Training and Testdata

# Get the feature data from EdgeImpulse
GET_FEATURES_FROM_EDGE = False
if GET_FEATURES_FROM_EDGE:
    API_KEY = 'ei_7c14274991477b16092db6eb8032e84b9f73c9c9e665b5c8bec01333b86151f3'
    X = (requests.get('https://studio.edgeimpulse.com/v1/api/18137/training/6/x', headers={'x-api-key': API_KEY})).content
    Y = (requests.get('https://studio.edgeimpulse.com/v1/api/18137/training/6/y', headers={'x-api-key': API_KEY})).content
    with open('x_train.npy', 'wb') as file:
        file.write(X)
    with open('y_train.npy', 'wb') as file:
        file.write(Y)

X = np.load('x_train.npy')
Y = np.load('y_train.npy')[:, 0]

classes_values = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "N", "o", "s"]
classes = len(classes_values)

Y = tf.keras.utils.to_categorical(Y - 1, classes)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

input_length = X_train[0].shape[0]

g_train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
g_validation_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))

# HP Params
HP_NEURONS_1 = hp.HParam('neurons_1', hp.Discrete([13]))
HP_NEURONS_2 = hp.HParam('neurons_2', hp.Discrete([23]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.2, 0.5))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd', 'rmsprop']))
HP_TRAINING_RATE = hp.HParam('training', hp.Discrete([0.00075]))
METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[HP_NEURONS_1,HP_NEURONS_2, HP_TRAINING_RATE],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')], )


def train_test_model(hparams):
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
    model.add(Reshape((int(input_length / 18), 18), input_shape=(input_length,)))
    model.add(Conv1D(hparams[HP_NEURONS_1] , kernel_size=3, activation='relu', padding='same')) #19 in current, best = 20 from Hyperparameter
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    model.add(Dropout(0.5))
    model.add(Conv1D(hparams[HP_NEURONS_2], kernel_size=3, activation='relu', padding='same')) #35, 24+16=40
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(classes, activation='softmax', name='y_pred'))

    # this controls the learning rate
    opt = Adam(lr=hparams[HP_TRAINING_RATE], beta_1=0.9, beta_2=0.999)
    # this controls the batch size, or you can manipulate the tf.data.Dataset objects yourself
    BATCH_SIZE = 32
    train_dataset, validation_dataset = set_batch_size(BATCH_SIZE, g_train_dataset, g_validation_dataset)

    # train the neural network
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    metrics_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir,
                                                      histogram_freq=1,
                                                      write_graph=True,
                                                      write_images=True,
                                                      update_freq='epoch',
                                                      profile_batch=2,
                                                      embeddings_freq=1)

    confusion_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)
    callbacks = [metrics_callback,confusion_callback]
#    callbacks=[]

    model.fit(train_dataset, epochs=140, validation_data=validation_dataset, verbose=2, callbacks=callbacks)
    loss, accuracy = model.evaluate(X_test, Y_test)

    return accuracy


# Run an experiment with HPPARAMS
def experiment(experiment_dir, hparams):
    with tf.summary.create_file_writer(experiment_dir).as_default():
        hp.hparams(hparams)
        accuracy = train_test_model(hparams)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)


# --------------------------------------------------------------------------------------
# Mainloop for experiments
# --------------------------------------------------------------------------------------

experiment_no = 0

for num_units_1 in HP_NEURONS_1.domain.values:
    for num_units_2 in HP_NEURONS_2.domain.values:
        for scales in HP_TRAINING_RATE.domain.values:
            hparams = {
                HP_NEURONS_1: num_units_1,
                HP_NEURONS_2: num_units_2,
                HP_TRAINING_RATE: scales
            }

            experiment_name = f'Experiment {experiment_no}'
            print(f'Starting Experiment: {experiment_name}')
            print({h.name: hparams[h] for h in hparams})
            experiment('logs/hparam_tuning/' + experiment_name, hparams)
            experiment_no += 1

print(logdir)
print("TensorFlow version: ", tf.__version__)
