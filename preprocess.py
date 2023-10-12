import os
import pandas as pd
import numpy as np
from pydub import AudioSegment
import librosa

fileDirMER = 'relaxed'
dataset = os.listdir(fileDirMER)
filewithdir = []
file = []

for audio in dataset:
    music_path = os.path.join(fileDirMER,audio)
    if(audio[-3:]=='wav'):
        fn=os.path.join(fileDirMER,audio)
        y,sr = librosa.load(fn)
        duration = librosa.get_duration(y=y,sr=sr)
        if duration > 25.5:
            filewithdir.append(fn)
            file.append(os.path.basename(fn))

path25 = "gen_relaxed/" #output dir

for i in range(len(filewithdir)):
    song = AudioSegment.from_mp3(filewithdir[i])  
    # cut to the audio clips to 25.5s
    new = song[:25500]
    path_new = path25 + os.path.basename(filewithdir[i])
    new.export(path_new,format = "mp3") 
