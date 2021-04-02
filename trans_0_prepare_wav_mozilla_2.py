#copies an even distributed number of samples from a folder into target folder

import csv
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub.silence import detect_silence
import numpy as np
from pydub.playback import play
import io
import os
import shutil
import random
import string
import random
from datetime import datetime
import constants as c
from shutil import copyfile

SOURCE_PATH = os.path.join(c.DATA_FOLDER, "wav_test")
TARGET_PATH = os.path.join(c.DATA_FOLDER, "wav_ffc_test")
NUMBER_OF_SAMPLES_PER_CLASS = 100
SHUFFLE_LIST = 1

# Build list of files in directory, array with first column is label
file_list = np.empty((0, 2))  # feature labels

for entry in os.scandir(SOURCE_PATH):
    label = entry.name.split('_')[0]

    if label in c.RESULT_CLASSES:
        file_member = np.array([label, entry.name])
        file_list = np.row_stack((file_list, file_member))
#        file_list = np.append(file_list, [file_member], axis=0)

# Check, that enough samples are there
r = np.asarray(np.unique(file_list[:, 0], return_counts=True)).transpose()
print("Occurance of values:")
print(r)
if (r[:, 1].astype(int) < NUMBER_OF_SAMPLES_PER_CLASS).all():
    print("WARNING: Not enough samples per class")
    exit()

#Start the copy
# Shuffle the list

if SHUFFLE_LIST==1:
    np.random.shuffle(file_list)

ri=np.zeros(r.shape[0], dtype=int)
rl=r[:,0]


##
# for i in range(file_list.shape[0]):

for f in file_list:
    label=f[0]
    index=r[:,0].tolist().index(label)
    if ri[index]<NUMBER_OF_SAMPLES_PER_CLASS:
        f_name=f[1]
        f_new_name=rl[index]+"."+str(ri[index])+"-"+f_name
        copyfile(os.path.join(SOURCE_PATH, f[1]),os.path.join(TARGET_PATH,f_new_name))
        ri[index]=ri[index]+1

print("DONE")



