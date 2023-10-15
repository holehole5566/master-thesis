import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy import stats

sheets = ["tempo", "rms_mean", "stft_pitches_mean", "zcr_mean"]
mood_list = ["happy", "relaxed", "sad", "angry"]

def plot(mood):

    for sheet in sheets:
        df = pd.read_excel("gen_" + mood + ".xlsx", sheet_name = sheet)
        series = df[sheet]
        '''
        series, lam = stats.boxcox(df[sheet])
        series = pd.DataFrame(series, columns=[sheet])[sheet]
        '''
        skewness = round(series.skew(), 2)
        kurtosis = round(series.kurt(), 2)
        res = stats.shapiro(series)
        # plot
        sns.histplot(series, kde=True)
        plt.title(f"Skewness: {skewness}, Kurtosis: {kurtosis}, Shapiro: {res.pvalue:.6f}" )
        plt.savefig(mood + "_"+ sheet + "_boxcox_remove_outlier.png")
        plt.clf()

def plot_all():
    for mood in mood_list:
        plot(mood)
