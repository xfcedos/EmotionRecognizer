import librosa
import librosa.display
import numpy as np
import os
import shutil
import math
import joblib
from keras.utils import to_categorical
from IPython.display import clear_output, display


almost_every_augmentation = [
    ('shift_by_second', {'shift_max': 2}), 
    ('noise_injection', {'noise_factor' : 0.4}),
    ('pitch_change', {'pitch_max' : 5, 'pitch_min' : -5})
]


def progress(idx, detail=None):
    clear_output(wait=True)
    if detail is None:
        display("Done: " + str(idx + 1))
    else:
        display(detail + " done: " + str(idx + 1))

    
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
    def __init__(self, path=None, amount=None, shuffle=False):
        self.path = path
        self.data = None
        self.labels = None
        if self.path is not None:
            self.path = path if path[-1] == '/' else path + '/'
            self.filelist = np.array(os.listdir(os.path.abspath(self.path)))
            if shuffle:
                idxs = np.random.permutation(len(self.filelist))
                self.filelist = self.filelist[idxs]
            self.filelist = self.filelist[:amount]
            
        
    def load_labels(self, to1hot=False):
        labels = []
        for file in self.filelist:
            labels.append([int(file[6:8]) - 1])
        
        self.labels = np.array(labels)
        if to1hot:
            self.labels = to_categorical(labels,num_classes=8)
        return self.labels
    
    def load_data(self, sr=22050):
        if self.path is None:
            raise NameError("No Path")

        data_list = []
        for idx, file in enumerate(self.filelist):
            x, s = librosa.core.load(self.path + file, res_type='kaiser_fast')
            data_list.append(x)
            progress(idx, 'Loading')
        self.data = data = np.array(data_list)
        print('Load complete')
        return self.data
    
    def load_dataset(self, sr=22050, to1hot=False):
        return self.load_data(sr=sr), self.load_labels(to1hot=to1hot)

    def save_via_joblib(self, external_data=(None, None), save_dir='./', names=('X', 'y')):
        

            
        if external_data[0] is not None:
            self.data = external_data[0]
            
        if external_data[1] is not None:
            self.labels = external_data[1]
        
        if self.data is None :
            raise NameError("Didn't load data")
            
        if self.labels is None:
            raise NameError("Didn't load labels")
            
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

    
def get_mfcc(data, sampling_rate, n_mfcc=40, conv_shaping=False):
    mfcc_array = []
    for idx, inst in enumerate(data):
        mfcc = np.mean(librosa.feature.mfcc(y=inst, sr=sampling_rate, n_mfcc=n_mfcc).T, axis=0)
        mfcc_array.append(mfcc)
        progress(idx)
    if conv_shaping:
        return np.reshape(np.array(mfcc_array), (len(data), n_mfcc, 1))
    return np.array(mfcc_array)


class MFCC_Prep:
    def __init__(self, sampling_rate=22050, n_mfcc=40):
        self.sampling_rate = sampling_rate
        self.n_mfcc = n_mfcc

            
    def augment_separately(self, data, labels, augmentations=[]):
        
        def get_mfcc(data):
            return np.mean(librosa.feature.mfcc(y=data, sr=self.sampling_rate, n_mfcc=self.n_mfcc).T,axis=0)

        def noise_injection(data, noise_factor):
            noise = np.random.randn(len(data))
            augmented_data = data + noise_factor * noise / 100
            augmented_data = augmented_data.astype(type(data[0]))
            return augmented_data

        def shift_by_proportion(data, shift_max, shift_direction='both'):

            shift = np.random.randint(int(shift_max * data.shape[-1]))
            if shift_direction == 'right':
                shift = -shift
            elif shift_direction == 'both':
                direction = np.random.randint(0, 2)
                if direction == 1:
                    shift = -shift    
            augmented_data = np.roll(data, shift)
            if shift > 0:
                augmented_data[:shift] = 0
            else:
                augmented_data[shift:] = 0
            return augmented_data


        def shift_by_second(data, shift_max, sampling_rate, shift_direction='both'):
            shift = np.random.randint(sampling_rate * shift_max)
            if shift_direction == 'right':
                shift = -shift
            elif shift_direction == 'both':
                direction = np.random.randint(0, 2)
                if direction == 1:
                    shift = -shift    

            augmented_data = np.roll(data, shift)
            if shift > 0:
                augmented_data[:shift] = 0
            else:
                augmented_data[shift:] = 0
            return augmented_data

        def pitch_change(data, pitch_max, pitch_min = 0):
            pitch_factor = np.random.randint(low=pitch_min, high=pitch_max)
            return librosa.effects.pitch_shift(data, self.sampling_rate, pitch_factor)

        '''
        def trim(data):
            return librosa.effects.trim(data)
        '''

        def stretch_to_size(data, expected_duration):
            rate = len(data) / (expected_duration * self.sampling_rate)
            return librosa.effects.time_stretch(data, rate)
        
        augmented_data = get_mfcc(data[0])
        augmented_labels = np.copy(labels)
        
        for idx, inst in enumerate(data[1:]):
            augmented_data = np.vstack((augmented_data, get_mfcc(inst)))
            progress(idx+1, 'copying')
        
        for (f, args) in augmentations:
            func = locals()[f]
            
            for idx, inst in enumerate(data):
                try:
                    augmented_inst = func(data=inst, **args)
                except TypeError:
                    augmented_inst = func(data=inst, sampling_rate=self.sampling_rate, **args)
                augmented_data = np.vstack((augmented_data, get_mfcc(augmented_inst)))
                progress(idx, func.__name__)
            augmented_labels = np.concatenate((augmented_labels, labels))
        return augmented_data, augmented_labels
