#Thiss predictor uses the trained model to predict the gender of an audio recording

from array import array
from struct import pack
from sys import byteorder
import copy
import pyaudio
import wave
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

########################## AUDIO RECORDER
THRESHOLD = 500  # audio levels not normalised.
CHUNK_SIZE = 1024
SILENT_CHUNKS = 3 * 44100 / 1024  # about 3sec
FORMAT = pyaudio.paInt16
FRAME_MAX_VALUE = 2 ** 15 - 1
NORMALIZE_MINUS_ONE_dB = 10 ** (-1.0 / 20)
RATE = 44100
CHANNELS = 1
TRIM_APPEND = RATE / 4

def is_silent(data_chunk):
    """Returns 'True' if below the 'silent' threshold"""
    return max(data_chunk) < THRESHOLD

def normalize(data_all):
    """Amplify the volume out to max -1dB"""
    # MAXIMUM = 16384
    normalize_factor = (float(NORMALIZE_MINUS_ONE_dB * FRAME_MAX_VALUE)
                        / max(abs(i) for i in data_all))

    r = array('h')
    for i in data_all:
        r.append(int(i * normalize_factor))
    return r

def trim(data_all):
    _from = 0
    _to = len(data_all) - 1
    for i, b in enumerate(data_all):
        if abs(b) > THRESHOLD:
            _from = max(0, i - TRIM_APPEND)
            break

    for i, b in enumerate(reversed(data_all)):
        if abs(b) > THRESHOLD:
            _to = min(len(data_all) - 1, len(data_all) - 1 - i + TRIM_APPEND)
            break

    return copy.deepcopy(data_all[_from:(_to + 1)])

def record(t):
    """Record a word or words from the microphone and 
    return the data as an array of signed shorts."""

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, output=True, frames_per_buffer=CHUNK_SIZE)

    silent_chunks = 0
    audio_started = False
    data_all = array('h')
    while True:
        # little endian, signed short
        data_chunk = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            data_chunk.byteswap()
        data_all.extend(data_chunk)

        silent = is_silent(data_chunk)

        if audio_started:
            
            if silent:
                silent_chunks += 1
                if silent_chunks > SILENT_CHUNKS:
                    break
            elif time.time()-start >= t:
                break
            else: 
                silent_chunks = 0
        elif not silent:
            audio_started = True              
            start = time.time()

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    data_all = trim(data_all)  # we trim before normalize as threshhold applies to un-normalized wave (as well as is_silent() function)
    data_all = normalize(data_all)
    return sample_width, data_all

def record_to_file(path, t):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record(t)
    data = pack('<' + ('h' * len(data)), *data)

    wave_file = wave.open(path, 'wb')
    wave_file.setnchannels(CHANNELS)
    wave_file.setsampwidth(sample_width)
    wave_file.setframerate(RATE)
    wave_file.writeframes(data)
    wave_file.close()

###################
#       END AUDIO RECORDER
##################


secs = int(input("How many seconds do you want to record?: "))
path = input("Input file name: ") + '.wav'
#print("Wait in silence to begin recording; wait in silence to terminate")
print("Recording...")
record_to_file("Data/" + path, secs)
print("done - result written to " + path)


f = open("trained_classifier", "rb")
rfc = pickle.load(f)


print("Measuring acoustic parameters...")
os.system("Rscript wav_analyzer.r >/dev/null 2>&1")
print("Done.")


df_pred = pd.read_csv('testData.csv')
#Reads audio data from testData
df_names = df_pred['sound.files'].copy()
#Saves names column to display them later
df_pred.drop(['sound.files', 'duration', 'selec', 'peakf', 'Unnamed: 0'], axis=1, inplace=True) #inplace=True
#Leaves only the columns that matter to make the prediction
prediction = rfc.predict(df_pred)
#Creates an array of ints that represent the predicted gender of each row of the dataFrame

print("-"*10)
for i in range(len(prediction)):
    #Displays the predicted genders in a formatted way
    print(df_names[i].split('.')[0].title(), "is a ", end='')
    if prediction[i] == 0:
        print("male")
    else:
        print("female")
