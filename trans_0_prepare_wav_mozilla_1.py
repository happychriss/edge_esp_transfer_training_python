# Extracts 600ms wave files from mozilla common voice database loaded in clips folder
# results is a list of files, one per wave in wav folders
# https://commonvoice.mozilla.org/en


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


CLIPS_FOLDER = os.path.join(c.DATA_FOLDER, "clips")

TYPE='train' # train or test
TSV_FILE = os.path.join(c.DATA_FOLDER,TYPE+".tsv")
TARGET_PATH = os.path.join(c.DATA_FOLDER,"wav_"+TYPE)

RESULT_SET = c.RESULT_CLASSES
INPUT_SET= c.RESULT_CLASSES_TXT

SAMPLE_SAVE = 30
SAMPLE_LENGTH = 700  # ms
SAMPLE_SHIFT = 100 # Window
MIN_VOICE_LENGHT = 170  # minimum lenght of a number (1,2,4)
MAX_VOICE_LENGTH = 800  # maximum length (7) // used as check
RUN_NAME = "shift"

NUMBER_OF_FILES = 50000

def random_code():
    tmp = ''.join((random.choice(string.ascii_letters + string.digits) for i in range(5)))
    return tmp

with open(TSV_FILE, newline='') as csvfile:
    file=csv.reader(csvfile, delimiter='\t', quotechar='"')
    next(file,None)
    no_of_files=0
    for row in file:

        no_of_files=no_of_files+1
        if no_of_files>NUMBER_OF_FILES:
            break

        label=row[2]
        filename=row[1]
        if label in INPUT_SET:
            sound = AudioSegment.from_mp3(os.path.join(CLIPS_FOLDER, filename))
            chunks = detect_silence(
                sound,
                min_silence_len=100,
                silence_thresh=-41,                 # anything under -16 dBFS is considered silence
            )
            if len(chunks)==2:
                silent_start = chunks[0][0] + SAMPLE_SAVE
                sound_start = chunks[0][1] - SAMPLE_SAVE
                sound_end = chunks[1][0] + SAMPLE_SAVE
                silent_end = chunks[1][1] - SAMPLE_SAVE
                if  MIN_VOICE_LENGHT <= sound_end-sound_start <= MAX_VOICE_LENGTH:
                    my_filename = RESULT_SET[INPUT_SET.index(label)]+"."+ RUN_NAME + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + random_code()

                    sample=sound[sound_start:sound_start+SAMPLE_LENGTH]
                    sample = sample.set_frame_rate(16000)
                    sample.export(os.path.join(TARGET_PATH, my_filename+"_N.wav"), format="wav")

                    sample=sound[sound_start+SAMPLE_SHIFT:sound_start+SAMPLE_LENGTH+SAMPLE_SHIFT]
                    sample = sample.set_frame_rate(16000)
                    sample.export(os.path.join(TARGET_PATH,  my_filename+"_S.wav"), format="wav")

                    print(".")
                else:
                    print("Invalid sound lenght: "+ str(sound_end-sound_start) + "---" + str(chunks) +" for file: "+filename)
                    #play(sound)
            else:
                print("Invalid chunks: " + str(chunks) + " for file: " + filename)
                #play(sound)
