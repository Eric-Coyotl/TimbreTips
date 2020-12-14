# -*- coding: utf-8 -*-
"""
@author: coyote
"""

import os
import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt
import wave
import sounddevice as sd
import soundfile as sf
from pydub import AudioSegment
from scipy import stats

def getFrequency(note, A4=440):
    notes = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
    
    # Ex : A3 vs C#5
    octave = int(note[2]) if len(note) == 3 else int(note[1])
    
    
    keyNumber = notes.index(note[0:-1]);
    
    if (keyNumber < 3) :
        keyNumber = keyNumber + 12 + ((octave - 1) * 12) + 1; 
    else:
        keyNumber = keyNumber + ((octave - 1) * 12) + 1; 

    return A4 * 2** ((keyNumber - 49) / 12)


class WaveObject(object):
    #file = file path
    def __init__(self, file, note):
        
        '''Frame and Hop size needed for spectrogram functions'''
        self.FRAME_SIZE = 2648   
        self.HOP_SIZE = 512
        
        
        self.file_name = file

        
        ipd.Audio(self.file_name)
        
        print(type(self.file_name))
        self.wavToArray, self.sr = librosa.load(self.file_name)


        ''' Creates ndarray object representing Spectrogram '''
        S_scale = librosa.stft(self.wavToArray, n_fft=self.FRAME_SIZE, hop_length = self.HOP_SIZE)
        S_scale.shape
        #type(S_scale[0][0])
        
        '''Calculate Spectrogram'''
        self.Y_scale = np.abs(S_scale) ** 2
        self.Y_log_scale = librosa.power_to_db(self.Y_scale)
        
        self.pitchFreq = round(getFrequency(note), 2)


    '''Plotting the waveform'''
    def visualize(self, plt_num: int): 
        # reading the audio file 
        raw = wave.open(self.file_name) 
          
        # reads all the frames  
        # -1 indicates all or max frames 
        signal = raw.readframes(-1) 
        signal = np.frombuffer(signal, dtype ="int16") 
          
        # gets the frame rate 
        f_rate = raw.getframerate() 
      
        # to Plot the x-axis in seconds  
        # you need get the frame rate  
        # and divide by size of your signal 
        # to create a Time Vector  
        # spaced linearly with the size  
        # of the audio file 
        
        time_in_seconds = len(signal) / f_rate
        
        time = np.linspace( 
            0, # start 
            time_in_seconds, 
            num = len(signal)
        ) 
    
        # using matplotlib to plot 
        # creates a new figure 
        plt.figure(plt_num) 
          
        # title of the plot 
        plt.title("Sound Wave") 
          
        # label of x-axis 
        plt.xlabel("Time") 
         
        # actual ploting 
        plt.plot(time, signal) 
        
        #Plot only a few periods of the waveform
        plt.xlim( (time_in_seconds/2) - (5/self.pitchFreq)  , (time_in_seconds/2) + (5/self.pitchFreq) )
        # shows the plot  
        # in new window 
        plt.show() 
      
        # you can also save 
        # the plot using 
        # plt.savefig('filename') 


    ''' plots the spectrogram '''
    def plot_spectrogram(self, y_axis = "linear"):
        if y_axis == "log":
            Y = self.Y_log_scale
        else:
            Y = self.Y_scale
        
       
        plt.figure(figsize=(25,10))
        librosa.display.specshow(Y,
                                 sr=self.sr,
                                 hop_length=self.HOP_SIZE,
                                 x_axis="time",
                                 y_axis=y_axis)
        plt.colorbar(format="%+2.f")
        
        
    def plot_chromagram(self):
        
        Y = self.wavToArray
        
        chroma = librosa.feature.chroma_cqt(y=Y, sr=self.sr)
        fig, ax = plt.subplots()
        img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
        ax.set(title='Chromagram demonstration')
        fig.colorbar(img, ax=ax)
        
        
        
        
    '''record your own wav file, default length is 30 sec '''
    
def recordWav(duration = 30):
    samplerate = 48000  # Hertz

    fileDir = os.path.dirname(os.path.realpath('__file__'))
    filename = os.path.join(fileDir, 'audio/current.wav')
       
    mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,
                        channels=2, blocking=True)
    sf.write(filename, mydata, samplerate)
        

def convertToMono(file_path_str):
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    filename = os.path.join(fileDir, 'audio/mono.wav')
    
    sound = AudioSegment.from_wav(file_path_str)
    sound = sound.set_channels(1)
    sound.export(filename, format="wav")
        