import os
import pandas as pd
import numpy as np
from pydub import AudioSegment
import librosa

def feature_1d(file):
    try:
        y, sr = librosa.load(file)
    except:
        print('No such file')
        quit()
    
    f = [] 
    
    # tempo
    onset_env = librosa.onset.onset_strength(y = y, sr = sr)
    dtempo = librosa.feature.tempo(onset_envelope = onset_env, sr = sr,
                               aggregate = None)
    tempo_mean = np.mean(dtempo)
    f.append(tempo_mean)  
    
    # RMS
    rms = librosa.feature.rms(y = y)
    rms_mean = np.mean(rms)
    rms_var = np.var(rms)
    f.append(rms_mean)
    f.append(rms_var)
    
    # pitches range(C3, C7)
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, fmin=150, fmax=2000)
    # get indexes of the maximum value in each time slice
    max_indexes = np.argmax(magnitudes, axis=0)
    # get the pitches of the max indexes per time slice
    pitches = pitches[max_indexes, range(magnitudes.shape[1])]
    
    pitches_mean = np.mean(pitches)
    pitches_var = np.var(pitches)
    f.append(pitches_mean)
    f.append(pitches_var)


    # zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y = y)
    zcr_mean = np.mean(zcr)
    zcr_var = np.var(zcr)
    f.append(zcr_mean)
    f.append(zcr_var)    

    return np.array(f)

writer = pd.ExcelWriter("gen_relaxed.xlsx", engine = 'xlsxwriter')

dir25 = "gen_music/relaxed"
data25 = os.listdir(dir25)
file_dir25 = []
count = 0
for audio in data25:
    count += 1
    if(audio[-3:] =='wav'):
        fn=os.path.join(dir25,audio)
        file_dir25.append(fn)
        print(count, fn)
size = count
features1 = np.zeros((size,7))
count = 0
for i in range(0,size):
    count += 1
    features1[i] = feature_1d(file_dir25[i])
    print(count, "finished")

feature1 = pd.DataFrame()
col_name = ['tempo','rms_mean','rms_var','stft_pitches_mean','stft_pitches_var','zcr_mean','zcr_var']
mood = []
for i in range(len(file_dir25)):
    mood.append("angry")

feature1['file'] = file_dir25
feature1['mood'] = mood
for i in range(len(col_name)):
    feature1[col_name[i]] = features1[:, i]

feature1.to_excel(writer, sheet_name = 'all')

for col in col_name:
    feature = feature1[col]
    feature = feature.to_frame()
    Q1 = feature[col].quantile(0.25)
    Q3 = feature[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    upper_array = np.where(feature[col]>=upper)[0]
    lower_array = np.where(feature[col]<=lower)[0]
    feature.drop(index=upper_array, inplace=True)
    feature.drop(index=lower_array, inplace=True)
    feature.to_excel(writer, sheet_name = col)
    
writer.close()