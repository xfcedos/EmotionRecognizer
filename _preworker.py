from IPython.display import clear_output, display
import librosa
import pandas as pd
import numpy as np
import joblib
import os

def progress(idx, detail=None):
    clear_output(wait=True)
    if detail is None:
        display("Done: " + str(idx + 1))
    else:
        display(detail + " done: " + str(idx + 1))

class Loader:
    def __init__(self, df_init, sr=22050, filename_col="filename"):
        self.filename_col = filename_col
        self.df_init = df_init
        self.sr = sr
        self.df_init.index = list(range(0, len(self.df_init)))
        
    def load(self, load_name = None):
        
        if load_name is None:
            data = list()

            for idx, row in self.df_init.iterrows():
                a_series, librosa_sr = librosa.core.load(row[self.filename_col], sr=self.sr, res_type="kaiser_fast")
                data.append(a_series)
                progress(idx, "Loading raw")
            self.data = np.array(data)
        else:
            self.data = joblib.load(load_name)
        
        return self.data
    
    def get_data(self, classes=None, label_only=False, label_col="emotion"):
        data = list()
        labels = list()
        classes = classes if classes is not None else self.df_init[label_col].values
        self.df_end = pd.DataFrame(columns=self.df_init.columns)
        
        for idx, row in self.df_init.iterrows():
            if row[label_col] in classes:
                if not label_only:
                    a_series = self.data[idx]
                    data.append(a_series)
                labels.append(row[label_col])
                self.df_end = self.df_end.append(row)
                    
        
        self.fdata = data = np.array(data)
        self.flabels = labels = np.array(labels)
        
        self.df_end.index = list(range(0, len(self.df_end)))
        
        if label_only:
            return labels
        
        return data, labels
    
    def get_train_test_split_per_actor(self, val_actors=[], test_actors=[]):
        d = {ds : [[], []] for ds in ["train", "test", "val"]}
        for idx, row in self.df_end.iterrows():
            if row["actor"] in val_actors:
                d["val"][0].append(self.fdata[idx])
                d["val"][1].append(self.flabels[idx])
            if row["actor"] in test_actors:
                d["test"][0].append(self.fdata[idx])
                d["test"][1].append(self.flabels[idx])
            else:
                d["train"][0].append(self.fdata[idx])
                d["train"][1].append(self.flabels[idx])
        for (k, v) in d.items():
            d[k][0] = np.array(d[k][0])
            d[k][1] = np.array(d[k][1])
        return d
    


sampling_rate = 22050
def noise_injection(data, noise_factor):
        noise = np.random.randn(len(data))
        augmented_data = data + noise_factor * noise / 100
        augmented_data = augmented_data.astype(type(data[0]))
        return augmented_data

def pitch_change(data, pitch_max, pitch_min = 0):
    pitch_factor = np.random.randint(low=pitch_min, high=pitch_max)
    return librosa.effects.pitch_shift(data,  sampling_rate, pitch_factor)


def trim(data):
    return librosa.effects.trim(data)[0]


def stretch_to_size(data, expected_duration):
    rate = len(data) / (expected_duration *  sampling_rate)
    return librosa.effects.time_stretch(data, rate)

def xtrim(x, initial_db = 60, min_db = 14, step = 2, save_per = 0.8):
    db = initial_db
    x_ = librosa.effects.trim(x, top_db=db)[0]
    while save_per < (len(x_) / len(x)) and db > min_db:
        db -= step
        x_ = librosa.effects.trim(x, top_db=db)[0]
    return x_

def augment_full(X, y):
    data = list()
    labels = list()
    
    bg_files = os.listdir('./_background_noise_/')
    bg_list = list()
    
    for f in bg_files:
        bg_list.append(librosa.load('./_background_noise_/'+f)[0])
    for idx, (x, y) in enumerate(zip(X, y)):
        x = trim(x)
        data.append(noise_injection(x, 0.2))
        data.append(pitch_change(x, 5, pitch_min=-5))
        
        bg = bg_list[np.random.randint(6)]

        start_ = np.random.randint(bg.shape[0]-len(x))
        bg_slice = bg[start_ : start_+len(x)]
        wav_with_bg = x * np.random.uniform(0.9, 1.2) + bg_slice * np.random.uniform(0, 0.05)
        data.append(wav_with_bg)
        data.append(x)
        for i in range(4):
            labels.append(y)
        progress(idx, detail = "Augment")
    return np.array(data), np.array(labels)

def mfcc_frame(X, Y, length = 22050 * 2, step=1024, n_mfcc=40, dim = 1):
    fx = list()
    fy = list()
    
    for idx, (x, y) in enumerate(zip(X, Y)):
        if len(x) < length:
            xa = librosa.util.fix_length(x, length, mode="edge")
            fx.append(xa)
            fy.append(y)
        else:
            i = 0
            
            while i < len(x) - length:
                xa = x[i:length+i]
                xa = librosa.feature.mfcc(xa, n_mfcc=n_mfcc)
                if dim == 1:
                    xa = np.mean(xa.T,axis=0)
                fx.append(xa)
                fy.append(y)
                i += step
        progress(idx, detail ="Framing and mfccing")
    fx = np.array(fx)
    fy = np.array(fy)
    return fx, fy

def frame_data(X, Y, length = 38590, step=2205, min_step = 2205, max_length=3 * 22050, amount=None):
    fx = list()
    fy = list()
    
    for idx, (x, y) in enumerate(zip(X, Y)):
        if amount is not None:
            step = int((len(x) - length) // amount)
        
        
        if len(x) < length or len(x) > max_length or step < min_step:
            rate = length / len(x)
            xa = librosa.effects.time_stretch(x, rate)
            
            xa = librosa.util.fix_length(xa, length, mode="edge")
            
            fx.append(xa)
            fy.append(y)
        else:
            i = 0
                    
            while i < len(x) - length:
                xa = x[i:length+i]
                fx.append(xa)
                fy.append(y)
                i += step
        progress(idx, detail ="Framing")
        
    fy = np.array(fy)
    
    try:
        fx = np.array(fx)
    except MemoryError:
        return fx, fy
    
    return fx, fy

def mfcc_data(X, n_mfcc=40, dim = 1):
    fx = list()
    for idx, x in enumerate(X):
        x_a = librosa.feature.mfcc(x, n_mfcc=n_mfcc)
        if dim == 1:
            x_a = np.mean(x_a.T,axis=0)
        fx.append(x_a)
        progress(idx, detail = "MFCC")
    try:
        return np.array(fx)
    except MemoryError:
        return fx

    
def augment_and_convert_to_mfcc(X, y, n_mfcc=40, dim = 1, bg_noise_count = 2, save_dir=None):
    def mfcc_data(x):
        x_a = librosa.feature.mfcc(x, n_mfcc=n_mfcc)
        if dim == 1:
            x_a = np.mean(x_a.T,axis=0)
        return x_a
    
    data = list()
    labels = list()
    
    bg_files = os.listdir('./_background_noise_/')
    bg_list = list()
    
    count = 0
    
    for f in bg_files:
        bg_list.append(librosa.load('./_background_noise_/'+f, res_type="kaiser_fast")[0])
        
    for idx, (x, y) in enumerate(zip(X, y)):
        data.append(mfcc_data(noise_injection(x, 0.2)))
        data.append(mfcc_data(pitch_change(x, 5, pitch_min=-5)))
        
        bg = [bg_list[i] for i in np.random.randint(len(bg_files), size=(bg_noise_count))]
        
        for bg_item in bg:
            start_ = np.random.randint(bg_item.shape[0]-len(x))
            bg_slice = bg_item[start_ : start_+len(x)]
            wav_with_bg = x * np.random.uniform(0.9, 1.2) + bg_slice * np.random.uniform(0, 0.05)
            data.append(mfcc_data(wav_with_bg))
        data.append(mfcc_data(x))
        
        for i in range(3 + bg_noise_count):
            labels.append(y)
            count += 1
        progress(idx, detail = "Augment")
        
        if save_dir is not None:
            for i, (d, l) in enumerate(zip(data, labels)):
                number = count - (3 + bg_noise_count - i)
                name = f"{number}-{l}"
                joblib.dump(d, save_dir + name)
            data = list()
            labels = list()
        
        
    return np.array(data), np.array(labels)