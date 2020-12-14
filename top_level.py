# -*- coding: utf-8 -*-
"""
@author: coyote
"""

from WaveObject import *
import os
import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt


def main():
    
    #recordWav(5)

    wave1 = WaveObject("audio/tr_c4e.wav", "A#3")
                        
    wave1.visualize(1)
    
    wave1.plot_spectrogram(y_axis = "log") 
    wave1.plot_chromagram()
    
    
if __name__ == "__main__":
    main()