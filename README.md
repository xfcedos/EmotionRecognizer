# EmotionRecognizer
Tensorflow-based emotion recognizing tool.

## What it does?
First, it recognizes voice's gender, then it recognizes it's emotion.

## How to start
### Install required libraries first 
```
pip install librosa tensorflow numpy noisereduce scikit-learn
```
or install using `requirements.txt`
```
pip install -r requirements.txt
```
## Process of learning
### Framing
Used 1.75sec overlapping frames with step = 0.1sec. There are more details, but it's beyond scope of README (for more detail look at frame_data() method in _preworker.py)
### MFCC
Used librosa method, with n_mfcc = 80 (see more in mfcc_data() in _preworker.py)

## Results on test set
### Female voice
  ![Female voice results](https://raw.githubusercontent.com/xfcedos/EmotionRecognizer/master/images/female.png)

### Male voice
  ![Male voice results](https://raw.githubusercontent.com/xfcedos/EmotionRecognizer/master/images/male.png)

## Used
### Datasets
- [Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)](https://smartlaboratory.org/ravdess/)
- [Surrey Audio-Visual Expressed Emotion (SAVEE)](http://kahlan.eps.surrey.ac.uk/savee/)
- [Toronto emotional speech set (TESS)](https://tspace.library.utoronto.ca/handle/1807/24487)
- [Crowd-sourced Emotional Mutimodal Actors Dataset (CREMA-D)](https://github.com/CheyneyComputerScience/CREMA-D)
