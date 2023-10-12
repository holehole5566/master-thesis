import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy import special

def data_length_normalizer(gt_data, obs_data, bins = 100):

    if len(gt_data) == len(obs_data):
        return gt_data, obs_data 

    if (len(gt_data) > 20 * bins) and (len(obs_data) > 20 * bins):
        bins = 10 * bins 
    # convert into frequency based distributions
    gt_hist = plt.hist(gt_data, bins = bins)[0]
    obs_hist = plt.hist(obs_data, bins = bins)[0]

    return gt_hist, obs_hist

def softmax(vec):
    """
    convert an array of values into an array of probabilities.
    """
    return(np.exp(vec)/np.exp(vec).sum())

def cal_kl_div(gt_data, obs_data):
    gt_data = softmax(gt_data)
    obs_data = softmax(obs_data)
    kl_div = special.kl_div(gt_data, obs_data)
    kl_div = round(sum(kl_div),4)
    return kl_div

moods = ["happy", "relaxed", "sad", "angry"]
features = ["tempo", "rms_mean", "stft_pitches_mean", "zcr_mean"]
for feature in features:
    result = []
    for mood_or in moods:
        result_row = []
        for mood_gen in moods:
            
            gen_file = "gen_" + mood_gen + ".xlsx"
            original_file = mood_or + ".xlsx"

            df_original = pd.read_excel(original_file, sheet_name = feature)
            df_generated = pd.read_excel(gen_file, sheet_name = feature)
            
            gt_data, obs_data = data_length_normalizer(
                gt_data = df_original[feature], # ground truth data
                obs_data = df_generated[feature] # observed data
            )
           
            kl_div = cal_kl_div(gt_data, obs_data)
            
            result_row.append(kl_div)
        
        result.append(result_row)

    print(feature + ":")
    print(result)

    
        
   
    
