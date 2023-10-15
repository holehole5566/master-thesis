import os
import pandas as pd
import numpy as np
from pydub import AudioSegment
import librosa



mood_list = ["happy", "relaxed", "sad", "angry"]

def add_file(dataset, mood, fileDirMER):
    filewithdir = []
    for audio in dataset:
        if(audio[-3:]=='wav'):
            fn = os.path.join(fileDirMER, audio)
            y,sr = librosa.load(fn)
            duration = librosa.get_duration(y = y, sr = sr)
            if duration > 25.5:
                filewithdir.append(fn)
    return filewithdir

def cut_audio(filewithdir, mood):
    path = "generated/" + mood 
    for i in range(len(filewithdir)):
        song = AudioSegment.from_mp3(filewithdir[i])  
        # cut to the audio clips to 25.5s
        new = song[:25500]
        path_new = path + os.path.basename(filewithdir[i])
        new.export(path_new, format = "mp3") 

def preprocess():
    for mood in mood_list:
        fileDirMER = 'generated/' + mood
        dataset = os.listdir(fileDirMER)
        filewithdir = add_file(dataset, mood, fileDirMER)
        cut_audio(filewithdir, mood)
    
    






