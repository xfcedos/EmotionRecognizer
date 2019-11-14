import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import shutil
import math

from IPython.display import clear_output, display

def create_all_in_one_folder(raw_data_path, fname):
    RAW_DATA_PATH = raw_data_path
    ALL_IN_ONE_PATH = os.path.join(RAW_DATA_PATH, fname)

    if not os.path.exists(ALL_IN_ONE_PATH):
        os.mkdir(os.path.abspath(ALL_IN_ONE_PATH))

    for i in range(1, 25):
        actor_path = RAW_DATA_PATH
        if i < 10:
            actor_path = RAW_DATA_PATH + '/Actor_0' + str(i)
        else:
            actor_path = RAW_DATA_PATH + '/Actor_' + str(i)
        for file in os.listdir(os.path.abspath(actor_path)):
            shutil.copy(actor_path + '/' + str(file), os.path.abspath(ALL_IN_ONE_PATH))
    
    return ALL_IN_ONE_PATH

            
class Loader:
    def __init__(self, path, duration='max', sr=22050, ravdess_naming=True, shuffle=True, start=0, end=None, standartization=True, time_stretch=True):
        self.duration = duration
        self.path = path
        self.sr = sr
        self.time_stretch = time_stretch
        self.ravdess_naming = ravdess_naming
        self.shuffle = shuffle
        self.start = start
        self.end = end
        self.dictionary = {
            1 : 'neutral', 
            2 : 'calm', 
            3 : 'happy', 
            4 : 'sad', 
            5 : 'angry', 
            6 : 'fearful', 
            7 : 'disgust', 
            8 : 'surprised'
        }
        self.standartization = standartization
    
    
    def load_labels(self):
        if self.ravdess_naming:
            labels = []
            for file in self.filelist:
                labels.append([int(file[6:8])])
            return np.array(labels)
        return None
    
    def get_longest_duration(self):
        return max([librosa.core.get_duration(filename=os.path.join(self.path, file)) for file in self.filelist])
             
    
    def load_instances(self):
        filelist = np.array(os.listdir(os.path.abspath(self.path)))
        idxs = np.random.permutation(len(filelist)) if self.shuffle else np.arange(len(filelist))
        self.idxs = idxs = idxs[self.start:self.end]
        self.filelist = filelist = filelist[idxs]
        
        if self.duration == 'max':
            self.duration = self.get_longest_duration()
        
        data_list = []

        for idx, file in enumerate(filelist):
            x, s = librosa.core.load(os.path.join(self.path, file), duration=self.duration, sr=self.sr)
            if self.time_stretch:
                rate = len(x) / math.ceil(self.duration * self.sr)
                data_list.append(librosa.effects.time_stretch(x, rate))
            else:
                data_list.append(librosa.util.fix_length(x, math.ceil(self.sr * self.duration)))
            clear_output(wait=True)
            display("Done: " + str(idx + 1))

        data = np.array(data_list)
        
        
        #Double sure
        err = np.geterr()
        
        #Standardization
        if self.standartization:
            with np.errstate(all='ignore'):
                mean = data.mean(axis=0)
                data -= mean
                std = data.std(axis=0)
                data /= std
               
        self.data = data = np.nan_to_num(data)
        
        np.seterr(**err)
        
        return data
    
    def save_as_txt(self, fname, fmt = '%.8e'):
        if self.data is not None:
            np.savetxt(fname, np.hstack((self.load_labels(), self.data)), fmt=fmt)

            
class LoadMelSpec(Loader):
    #... for future purposes
    def __init__(self):
        pass
    
'''
    
##Examples

RAW_DATA_PATH = './datasets/RAVDESS/'

create_all_in_one_folder(RAW_DATA_PATH, "All")


loader = Loader(path)
X = loader.load_instances()
y = loader.load_labels()

'''
