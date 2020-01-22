import sys
import librosa
import keras
import numpy as np

from _preworker import *

emo_list = ["angry", "disgust", "fearful", "happy", "neutral", "sad"]

def main():
    if len(sys.argv) != 3:
        raise RuntimeError()
    file = sys.argv[1]
    gender = sys.argv[2].lower()
    data, sr_ = librosa.core.load(file)
    datalist, _ = frame_data([data], [None])
    datamfcc = mfcc_data(datalist)
    
    if gender == "male":
        model = keras.models.load_model("male.h5")
    elif gender == "female":
        model = keras.models.load_model("female.h5")
    else:
        raise Exception("no gender")
    
    frame_results = np.argmax(model.predict(datamfcc), axis=0)
    emotion = emo_list[np.argmax(np.bincount(frame_results))]
    print(emotion)
    return emotion

if __name__ == "main":
    main()