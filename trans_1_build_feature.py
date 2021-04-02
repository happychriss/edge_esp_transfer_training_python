#

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import os
from dsp import generate_features
from pydub import AudioSegment
import constants as c

SOURCE_WAVE_FOLDER = c.DATA_FOLDER
TARGET_FEATURE_FOLDER = c.FEATURES_FOLDER

# Which folder to use as input - files will be in feature folder with the same name
WAVES = 'wav_mytrain'

SOURCE_WAVE_FOLDER = os.path.join(SOURCE_WAVE_FOLDER, WAVES)
MAX_FILES = 50000

SOUND_LENGHT = 600  # length of clip used for feature cration
SOUND_SHIFT_WINDOW = 100

FEATURE_STRIDES = 46
FEATURE_HIGHT = 18
FEATURE_SIZE = FEATURE_HIGHT * FEATURE_STRIDES

RESULT_SET = c.RESULT_CLASSES
# WAVE_FILE = os.path.join(DATA_FOLDER, "8.v6_15_14_NO_CHANGE_20210217_222522_46Glw.wav")
# FEATURE_FILE = os.path.join(DATA_FOLDER, "8_features.txt")
# w = np.genfromtxt(FEATURE_FILE, delimiter=',')

X = np.empty((0, FEATURE_SIZE), np.float32)  # feature values
Y = np.empty(0, np.int32)  # feature labels
file_count = 0

for entry in os.scandir(SOURCE_WAVE_FOLDER):
    file_count = file_count + 1

    # reduce to 600ms, time window used in EdgeImpulse
    sound = AudioSegment.from_file(entry.path).set_frame_rate(16000)[0:600]
    if sound.duration_seconds == 0.6:

        # Convert from Wave file into RawData, 16bit signed, dtype='h'
        raw_data = np.frombuffer(sound.raw_data, dtype='h')

        feature_data = generate_features(implementation_version=2,
                                         draw_graphs=False,
                                         raw_data=raw_data,
                                         axes="x",
                                         sampling_freq=16000,
                                         frame_length=0.013,
                                         frame_stride=0.013,
                                         num_filters=32,
                                         fft_length=256,
                                         num_cepstral=18,
                                         win_size=261,
                                         low_frequency=300,
                                         high_frequency=8000,
                                         pre_cof=0.98,
                                         pre_shift=1)

        features = np.array(feature_data['features'], np.float32)
        X = np.append(X, [features], axis=0)
        Y = np.append(Y, [RESULT_SET.index(entry.name[0])], axis=0).astype(np.int32)

        if SOUND_SHIFT_WINDOW!=0:
            sound = AudioSegment.from_file(entry.path).set_frame_rate(16000)[100:700]
            if sound.duration_seconds == 0.6:
                # Convert from Wave file into RawData, 16bit signed, dtype='h'
                raw_data = np.frombuffer(sound.raw_data, dtype='h')

                feature_data = generate_features(implementation_version=2,
                                                 draw_graphs=False,
                                                 raw_data=raw_data,
                                                 axes="x",
                                                 sampling_freq=16000,
                                                 frame_length=0.013,
                                                 frame_stride=0.013,
                                                 num_filters=32,
                                                 fft_length=256,
                                                 num_cepstral=18,
                                                 win_size=261,
                                                 low_frequency=300,
                                                 high_frequency=8000,
                                                 pre_cof=0.98,
                                                 pre_shift=1)

                features = np.array(feature_data['features'], np.float32)
                X = np.append(X, [features], axis=0)
                Y = np.append(Y, [RESULT_SET.index(entry.name[0])], axis=0).astype(np.int32)

        if file_count == MAX_FILES:
            break
        print(".")

    else:
        print("warning: clip to short:" + str(sound.duration_seconds))

np.save(os.path.join(TARGET_FEATURE_FOLDER, "x_" + WAVES + ".npy"), X)
np.save(os.path.join(TARGET_FEATURE_FOLDER, "y_" + WAVES + ".npy"), Y)
print(X)
print(Y)
