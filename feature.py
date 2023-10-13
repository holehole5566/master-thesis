import os
import pandas as pd
import numpy as np
from pydub import AudioSegment
import librosa

col_name = ['tempo','rms_mean','rms_var','stft_pitches_mean','stft_pitches_var','zcr_mean','zcr_var']
mood_list = ["happy", "relaxed", "sad", "angry"]

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

def get_music_files(dir):
    data = os.listdir(dir)
    file_dir = []
    count = 0
    for audio in data:
        count += 1
        if(audio[-3:] == 'wav' or audio[-3:] == 'mp3'):
            fn=os.path.join(dir,audio)
            file_dir.append(fn)
            print(count, fn)
    return file_dir, count

def extract_features(file_dir, size):
    features = np.zeros((size,7))
    for i in range(0, size):
        features[i] = feature_1d(file_dir[i])
    return features

def get_moods(size, mood):
    moods = []
    for i in range(len(file_dir)):
        moods.append(mood)
    return moods

def append_columns(all_features, file_dir, moods):
    all_features['file'] = file_dir
    all_features['mood'] = moods
    for i in range(len(col_name)):
        all_features[col_name[i]] = all_features[:, i]
    return all_features

def remove_outlier(feature, col):
    Q1 = feature[col].quantile(0.25)
    Q3 = feature[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    upper_array = np.where(feature[col]>=upper)[0]
    lower_array = np.where(feature[col]<=lower)[0]
    feature.drop(index=upper_array, inplace=True)
    feature.drop(index=lower_array, inplace=True)
    return feature

def write_each_feature(all_features, writer):
    for col in col_name:
        feature = all_features[col]
        feature = feature.to_frame()
        feature = remove_outlier(feature, col)
        feature.to_excel(writer, sheet_name = col)

def write_all_features():
    
    for mood in mood_list:

        writer = pd.ExcelWriter("gen_" + mood + ".xlsx", engine = 'xlsxwriter')

        dir = "gen_music/" + mood
    
        file_dir, size = get_music_files(dir)
    
        all_features = extract_features(file_dir, size)
        all_features = pd.DataFrame()
    
        moods = get_moods(len(file_dir), mood)
    
        all_features = append_columns(all_features, file_dir, moods)
    
        all_features.to_excel(writer, sheet_name = 'all')

        write_each_feature(all_features, writer)
    
        writer.close()



