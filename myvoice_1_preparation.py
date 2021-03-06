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

random.seed(a=None, version=2)

RUN_NAME = "validate"
SOURCE_FOLDER = "./data/clips_myvoice/"  # first character of filename = label, "G" use Google, X for ignore, N for noise
NOISE_SAMPLE = "./data/clips_myvoice/r_raw_speech20210319-212828.raw"
TARGET_FOLDER = "./data/wav_myvoice_validate"

VALIDATE_MODE=1 # if 1 = no split in training and test subfolder is done
SUBFOLDER_TRAINING = "train"
SUBFOLDER_TEST = "test"
TRAIN_PERCENTAGE = 0.8 # 0.8, means 80% training, 20% test


NUMBER_OF_VERSIONS = 4

NOISE_PERCENTAGE = 0.5

SAMPLE_LENGTH = 700  # ms
TIME_WINDOW = 150 # ms

# Plausibility checks, if not in this range, clip will be dropped
MIN_VOICE_LENGHT = 170  # minimum lenght of a number (1,2,4)
MAX_VOICE_LENGTH = 800  # maximum length (7) // used as check

SAMPLE_SAVE = 35  # to have some buffer on silent detection
MAX_SILENCE_WINDOW = 40  # maximum silence at the begin of a sound,  CLASSIFIER_SLICES_PER_MODEL_WINDOW

NUMBER_OF_NOISE_SAMPLES = 0 #90 # 240
noise = AudioSegment.from_raw(NOISE_SAMPLE, sample_width=2, frame_rate=16000, channels=1)
random.seed(a=None, version=2)

result_set = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0","j","n"]
LabelsList = []
vers_samples = []

class Original():
    "Stores name and place pairs"

    def __init__(self, silent_start, sound_start, sound_end, silent_end, label):
        self.silent_start = silent_start
        self.sound_start = sound_start
        self.sound_end = sound_end
        self.silent_end = silent_end
        self.label = label


class Versions():
    def __init__(self, original_idx, start, end, low_pass, db_voice, db_noise, label, voice_length, sound, file_idx):
        self.original_idx = original_idx
        self.start = start
        self.end = end
        self.low_pass = low_pass  # low pass used
        self.db_voice = db_voice  # change in DB used
        self.db_noice = db_noise
        self.label = label
        self.voice_length = voice_length
        self.audio = 0
        self.output_voice = sound[start:end]
        self.file_idx = file_idx

    def get_sound(self):
        if self.db_voice != 0:
            out_voice = (self.output_voice + self.db_voice).low_pass_filter(self.low_pass)
            noise_random_start = random.randrange(0, int(noise.duration_seconds * 1000)) - SAMPLE_LENGTH
            out_noise = noise[noise_random_start:noise_random_start + SAMPLE_LENGTH] + self.db_noice
            output = out_voice.overlay(out_noise)
        else:
            output = self.output_voice

        return output


# ## this was an idea to generate shorter files, as only "7" is very long and needs 900ms window
def split_seven_new_start(my_audio):
    # play(audio)
    steps_size = 30
    slice_starts = range(0, int(my_audio.duration_seconds * 1000), steps_size)
    min_silence_len = 1

    max_rms = 0
    for i in slice_starts:
        audio_slice = my_audio[i:i + steps_size]
        rms = audio_slice.rms
        if rms > max_rms:
            max_rms = rms
            max_sec = i

    print("Max RMS: " + str(max_rms))

    return int(max_sec)


def random_code():
    tmp = ''.join((random.choice(string.ascii_letters + string.digits) for i in range(5)))
    return tmp


def process_noise_file(my_filename):
    sound = AudioSegment.from_raw(my_filename, sample_width=2, frame_rate=16000, channels=1)
    print("************** Noise Samples **************")
    my_split = SUBFOLDER_TRAINING
    for i in range(1, NUMBER_OF_NOISE_SAMPLES):
        sound_start = random.randrange(0, (int(sound.duration_seconds * 1000)) - SAMPLE_LENGTH)
        my_noise = sound[sound_start:sound_start + SAMPLE_LENGTH] + random.randrange(-10, 2)
        my_filename = "N." + RUN_NAME + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + random_code() + ".wav"

        if i > int(NUMBER_OF_NOISE_SAMPLES * TRAIN_PERCENTAGE): my_split = SUBFOLDER_TEST
        my_filename = os.path.join(TARGET_FOLDER, my_split, my_filename)

        my_noise.export(out_f=my_filename, format='wav')
        print("************** DONE **************")


# Processes one sample file with multiple samples, extracts orginal sample and genearate multiple version of the samples
# that is attached to a global list

def process_file(my_label_name, my_filename, file_idx):
    sound = AudioSegment.from_raw(my_filename, sample_width=2, frame_rate=16000, channels=1)

    print("PHASE-1: ******************************************************** Silence Detection **************")

    chunks = detect_silence(
        sound,
        # split on silences longer than 1000ms (1 sec)
        min_silence_len=200,
        # anything under -16 dBFS is considered silence
        silence_thresh=-44,
    )

    print("Silence Chunks:")
    for o_idx, chunk in enumerate(chunks):
        print(str(o_idx) + ":" + str(chunk[0]) + " - " + str(chunk[1]))

    # Strip the audio by chunks and build an original number snippet for each number named

    print("PHASE-2: ******************************************************** Build Snippets  **************")

    # # build orginal number array #  [-1/0]ssssss[-1/1]mmmmmm[0/0]sssss[0/1]
    org_samples = []
    for o_idx, chunk in enumerate(chunks):
        if o_idx > 1:

            silent_start = chunks[o_idx - 1][0] + SAMPLE_SAVE
            sound_start = chunks[o_idx - 1][1] - SAMPLE_SAVE
            sound_end = chunks[o_idx][0] + SAMPLE_SAVE
            silent_end = chunks[o_idx][1] - SAMPLE_SAVE

            data_label = my_label_name

            if my_label_name == 'X': # handling number seven as special
                new_start = split_seven_new_start(sound[sound_start:sound_end]) + sound_start
                org_samples.append(Original(new_start - 210, new_start, sound_end, silent_end, data_label))
            #                play(sound[new_start-50:silent_end])
            else:
                org_samples.append(Original(silent_start, sound_start, sound_end, silent_end, data_label))

    print("Build Original for Label:" + initial_data_label + " Snippets(numbers) found: " + str(len(org_samples)))

    # For each sound snippet (org_samples), build multiple versions

    print("PHASE-3: ******************************************************** Build Versions  **************")

    for o_idx, o_sample in enumerate(org_samples):


        sound_length = o_sample.sound_end - o_sample.sound_start

        # Plausibility checks
        if sound_length < MIN_VOICE_LENGHT or sound_length > MAX_VOICE_LENGTH:
            print("Wave(" + str(o_idx) + "): ERROR - Sound length to small/big: " + str(sound_length))
            continue


        max_silent=random.randrange(0, MAX_SILENCE_WINDOW)
        if sound_length<400:
            max_silent=max_silent+40

        print("Wave(" + str(o_idx) + "): " + "-> " + str(max_silent) + "-" + str(sound_length))

        sound_new_start = o_sample.sound_start - max_silent
        sound_new_end = sound_new_start + SAMPLE_LENGTH

        # This version is the same as the original

        vers_samples.append(
            Versions(o_idx, sound_new_start, sound_new_end,
                     0,  # low_pass
                     0,  # db for voice - no change
                     0,  # db for noise is very low
                     o_sample.label,
                     sound_length, sound, file_idx))


        window_offset= random.sample(range(1, TIME_WINDOW), NUMBER_OF_VERSIONS)
        for i in range(1, NUMBER_OF_VERSIONS):

            shift = window_offset[i-1]

            sound_new_start = o_sample.sound_start + shift
            sound_new_end = sound_new_start + SAMPLE_LENGTH

            if random.random() > NOISE_PERCENTAGE:
                noise_db = random.randrange(-20, -5)
            else:
                noise_db = -100

            # modify samples with db, low_pass and noise
            vers_samples.append(
                Versions(o_idx, sound_new_start, sound_new_end,
                         random.randrange(500, 6000),  # low_pass
                         random.randrange(-5, 2),  # db for voice
                         noise_db,  # db for noise
                         o_sample.label,
                         sound_length, sound, file_idx))

    print("DONE")


# START PROGRAM **************************************************************************************************

# Prepare file list **************************************************************************************************
# G=Label via Google, X=ignore, file should end with raw
# All sound files are processed and different versions created for each snippet (e.g. nubmer) , added to globallist vers_samples

entry_idx = 0
for entry in os.scandir(SOURCE_FOLDER):

    initial_data_label = ""
    if entry.name.endswith(".raw") and entry.name[0].upper() != "X":
        if entry.name[0] in result_set:
            entry_idx += 1
            initial_data_label = str(entry.name[0])
            print("!!!!!!!!!!!!!!!!! Processing file:" + entry.name + " with label: " + initial_data_label + " !!!!!!!!!!!!!!!!!!!!")
            process_file(initial_data_label, entry.path, entry_idx)
        elif entry.name[0].upper() == "R":
            process_noise_file(entry.path)
        else:
            print("Skipped file:" + entry.name)

# Iterate over all labels and filter on vers_samples to write sample file
# Split between training and testdata

for label_idx, label_name in enumerate(result_set):

    v_labels_list = [x for x in vers_samples if x.label == label_name]
    SPLIT = SUBFOLDER_TRAINING

    number_of_labels = len(v_labels_list)
    number_of_training_samples = int(len(v_labels_list) * TRAIN_PERCENTAGE)

    # Iterate over all versions of a specific label

    avg_voice_length = 0

    if number_of_labels > 0:

        for version_idx, version in enumerate(v_labels_list):

            if version.db_voice == 0:
                filename = version.label + "." + RUN_NAME + "_" + str(version.file_idx) + "_" + str(version.original_idx) + "_NO_CHANGE" \
                           + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + random_code() + ".wav"
            else:
                filename = version.label + "." + RUN_NAME + "_" + str(version.file_idx) + "_" + str(version.original_idx) + "_DBS" + str(
                    version.db_voice) + "_DBN" + str(
                    version.db_noice) + "_LP" + str(
                    version.low_pass) + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + random_code() + ".wav"

            avg_voice_length = avg_voice_length + version.voice_length

            if VALIDATE_MODE==1:
                filename = os.path.join(TARGET_FOLDER, filename)
            else:
                if version_idx > number_of_training_samples: SPLIT = SUBFOLDER_TEST
                filename = os.path.join(TARGET_FOLDER, SPLIT, filename)

            version.get_sound().export(out_f=filename, format='wav')

        print("Label: " + label_name + " Samples: " + str(number_of_labels) + " Trainingssamples: " + str(
            number_of_training_samples) + " Avg. Voicelength: " + str(int(avg_voice_length / number_of_labels)))
