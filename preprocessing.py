import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import shutil
import math
import joblib

from IPython.display import clear_output, display

class Loader2:
    def __init__(self, path=None, duration='max', sr=22050, shuffle=True, start=0, end=None, mode='time_stretch'):
        self.duration = duration
        self.path = path
        self.sr = sr
        self.mode = mode #time_stretch, pad and both are possible 
        self.shuffle = shuffle
        self.start = start
        self.end = end
        self.data = None
        
        if path is not None:
            filelist = np.array(os.listdir(os.path.abspath(self.path)))
            idxs = np.random.permutation(len(filelist)) if self.shuffle else np.arange(len(filelist))
            self.idxs = idxs = idxs[self.start:self.end]
            self.filelist = filelist = filelist[idxs]
            if self.duration == 'max':
                self.duration = self.longest_duration()
            self.labels = self.get_labels()
    
    def longest_duration(self):
        if self.path is None:
            raise NameError("No path")
        return max([librosa.core.get_duration(filename=self.path +  file) for file in self.filelist])

    def get_labels(self):
        if self.path is None:
            raise NameError("No path")
        labels = []
        for file in self.filelist:
            labels.append([int(file[6:8]) - 1])
            if self.mode == 'both':
                labels.append([int(file[6:8]) - 1])
        return np.array(labels)
    
    def load(self):
        if self.path is None:
            raise NameError("No path")
        data_list = []

        for idx, file in enumerate(self.filelist):
            x, s = librosa.core.load(self.path + file, duration=self.duration, sr=self.sr, res_type='kaiser_fast')
            if self.mode == 'time_stretch':
                rate = len(x) / math.ceil(self.duration * self.sr)
                data_list.append(librosa.effects.time_stretch(x, rate))
            elif self.mode == 'pad':
                data_list.append(librosa.util.fix_length(x, math.ceil(self.sr * self.duration)))
            else:
                rate = len(x) / math.ceil(self.duration * self.sr)
                data_list.append(librosa.effects.time_stretch(x, rate))
                data_list.append(librosa.util.fix_length(x, math.ceil(self.sr * self.duration)))
            clear_output(wait=True)
            display("Done loading: " + str(idx + 1)) if self.mode != 'both' else display("Done loading: " + str(2 * idx + 2))
            
        self.data = data = np.array(data_list)
        if self.mode == 'both' and self.shuffle:
            extra_permutation = np.random.permutation(len(data_list))
            self.data = self.data[extra_permutation]
            self.labels = self.labels[extra_permutation]
        print('Load complete')
        return self.data, self.labels
    
    def get_MFCC(self, n_mfcc=40):
        if self.path is None:
            raise NameError("No path")
            
        if self.data is None:
            self.load()
        lst = []
        self.n_mfcc = n_mfcc
        for idx, x in enumerate(self.data):
            mfccs = np.mean(librosa.feature.mfcc(y=x, sr=self.sr, n_mfcc=n_mfcc).T,axis=0)
            lst.append(mfccs)
            clear_output(wait=True)
            display("Done MFCCing: " + str(idx + 1)) if self.mode != 'both' else display("Done loading: " + str(2 * idx + 2))
        print('MFCCing complete')
        return np.array(lst), self.labels

    def save_via_joblib(self, path='./', names=('X', 'y')):
        if self.path is None:
            raise NameError("No path")
        if self.data is None:
            raise NameError("Didn't load")
        if not os.path.exists(path):
            os.mkdir(os.path.abspath(path))
        joblib.dump(self.data, os.path.join(path, names[0]))
        joblib.dump(self.labels, os.path.join(path, names[1]))
    
    def load_via_joblib(self, path='./', names=('X', 'y')):
        pathX = path + names[0]
        pathY = path + names[1]
        self.data = joblib.load(pathX)
        self.labels = joblib.load(pathY)
    
    #works with fully used dataset only
    def shuffle_data_with_other_Loader(self, loader):
        if self.path is None:
            raise NameError("No path")
        adj_data = np.vstack((self.data, loader.data))
        adj_labels = np.vstack((self.labels, loader.labels))
        idxs = np.shuffle(len(adj_labels))
        return adj_data[idxs], adj_labels[idx]