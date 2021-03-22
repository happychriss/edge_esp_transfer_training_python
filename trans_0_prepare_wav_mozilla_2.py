
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


SOURCE_PATH = os.path.join(c.DATA_FOLDER,"wav_test")
TARGET_PATH= os.path.join(c.DATA_FOLDER,"edge_wav_test")
NUMBER_OF_SAMPLES_PER_CLASS=100

file_list= np.empty((0,2))  # feature labels

for entry in os.scandir(SOURCE_PATH):
    label=entry.name.split('_')[0]

    if label in c.RESULT_CLASSES:
        file_member=np.array([label,entry.name])
        file_list=np.row_stack((file_list, file_member))
#        file_list = np.append(file_list, [file_member], axis=0)

print(file_list)