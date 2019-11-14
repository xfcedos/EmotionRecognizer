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

def progress(idx, detail=''):
    clear_output(wait=True)
    display(detail + "Done: " + str(idx + 1))

    
class Loader:
    def __init__(self, path=None, start=0, amount=None, all_in_one_folder=False, all_in_one_folder_name='All'):
        self.path = path
        self.start = start
        self.end = amount + start if amount is not None else None
        self.data = None
        if path is not None:
            if not all_in_one_folder:
                self.create_all_in_one_folder(path, all_in_one_folder_name)
                self.path = path + all_in_one_folder_name + '/'
            self.filelist = np.array(os.listdir(os.path.abspath(self.path)))[self.start:self.end]
            self.labels = self.load_labels()
        self.decryptor = {
            0 : 'neutral', 
            1 : 'calm', 
            2 : 'happy', 
            3 : 'sad', 
            4 : 'angry', 
            5 : 'fearful', 
            6 : 'disgust', 
            7 : 'surprised'
        }
        
                

    def create_all_in_one_folder(self, raw_data_path, fname):
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
    
    def load_labels(self):
        labels = []
        for file in self.filelist:
            labels.append([int(file[6:8])])
        return np.array(labels)
        
    def load_instances_list(self): 
        if self.path is None:
            raise NameError("No path")
        data_list = []

        for idx, file in enumerate(self.filelist):
            x, s = librosa.core.load(self.path + file, res_type='kaiser_fast')
            data_list.append(x)
            progress(idx)
            
        self.data = data = np.array(data_list)
        print('Load complete')
        return self.data, self.labels

    def save_via_joblib(self, external_data=None, save_dir='./', names=('X', 'y')):
        if self.path is None:
            raise NameError("No path")
        if self.data is None:
            raise NameError("Didn't load")
        if external_data is not None:
            self.data = external_data
        if not os.path.exists(save_dir):
            os.mkdir(os.path.abspath(save_dir))
        joblib.dump(self.data, os.path.join(save_dir, names[0]))
        joblib.dump(self.labels, os.path.join(save_dir, names[1]))
    
    def load_via_joblib(self, load_dir='./', names=('X', 'y')):
        pathX = load_dir + names[0]
        pathY = load_dir + names[1]
        self.data = joblib.load(pathX)
        self.labels = joblib.load(pathY)
        return self.data, self.labels

def get_mfcc(data, sampling_rate, n_mfcc=40):
    mfcc_array = []
    for idx, inst in enumerate(data):
        mfcc = np.mean(librosa.feature.mfcc(y=inst, sr=sampling_rate, n_mfcc=n_mfcc).T, axis=0)
        mfcc_array.append(mfcc)
        progress(idx)
    return np.array(mfcc_array)

def noise_injection(data, noise_factor):
    augmented_data = list()
    for idx, inst in enumerate(data):
        noise = np.random.randn(len(inst))
        augmented_inst = inst + noise_factor * noise / 100
        augmented_inst = inst.astype(np.float32)
        augmented_data.append(augmented_inst)
        progress(idx)
    return np.array(augmented_data)

def shift_by_proportion(data, shift_max, shift_direction='both'):
    augmented_data = list()
    for idx, inst in enumerate(data):
        shift = np.random.randint(int(shift_max * len(inst)))
        if shift_direction == 'left':
            shift = -shift
        elif shift_direction == 'both':
            direction = np.random.randint(0, 2)
            if direction == 1:
                shift = -shift    
        augmented_inst = np.roll(inst, shift)
        if shift > 0:
            augmented_inst[:shift] = 0
        else:
            augmented_inst[shift:] = 0
        augmented_data.append(augmented_inst)
        progress(idx)
    return np.array(augmented_data)

def shift_by_second(data, shift_max, sampling_rate, shift_direction='both'):
    augmented_data = list()
    for idx, inst in enumerate(data):
        shift = np.random.randint(sampling_rate * shift_max)
        if shift_direction == 'left':
            shift = -shift
        elif shift_direction == 'both':
            direction = np.random.randint(0, 2)
            if direction == 1:
                shift = -shift    
        augmented_inst = np.roll(inst, shift)
        if shift > 0:
            augmented_inst[:shift] = 0
        else:
            augmented_inst[shift:] = 0
        augmented_data.append(augmented_inst)
        progress(idx)
    return np.array(augmented_data)

def pitch_change(data, sampling_rate, pitch_factor):
    augmented_data = list()
    for idx, inst in enumerate(data):
        augmented_data.append(librosa.effects.pitch_shift(inst, sampling_rate, pitch_factor))
        progress(idx)
    return np.array(augmented_data)

def trim(data):
    augmented_data = list()
    for idx, inst in enumerate(data):
        augmented_data.append(librosa.effects.trim(inst)[0])
        progress(idx)
    return np.array(augmented_data)

def stretch_to_size(data, sampling_rate, expected_duration='max'):
    if expected_duration == 'max':
        expected_len = max([len(inst) for inst in data])
    else:
        expected_len = expected_duration * sampling_rate
    augmented_data = list()
    for idx, inst in enumerate(data):
        rate = len(inst) / expected_len
        augmented_data.append(librosa.effects.time_stretch(inst, rate))
        progress(idx)
    return np.array(augmented_data)

def pad_to_size(data, sampling_rate, expected_duration='max'):
    augmented_data = list()
    if expected_duration == 'max':
        expected_len = max([len(inst) for inst in data])
    else:
        expected_len = sampling_rate * expected_duration
    for idx, inst in enumerate(data):
        augmented_data.append(librosa.util.fix_length(inst, expected_len))
        progress(idx)
    return np.array(augmented_data)
