from spectrogram import *
import cv2

import wave
import pylab


def graph_spectrogram(wav_file, spectogram):
    sound_info, frame_rate = get_wav_info(wav_file)
    pylab.figure(num=None, figsize=(19, 12))
    pylab.subplot(111)
    pylab.title('spectrogram of %r' % wav_file)
    pylab.specgram(spectogram, Fs=frame_rate)
    pylab.savefig(wav_file.split('.')[0]+'.png')


def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'int16')
    print(sound_info.shape)
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

wav_file_path = '/Users/aryaman/research/FER_datasets/video/ErinBrockavich_shot_4.wav'
spec_img = get_spectrum(wav_file_path)
spec_img = spec_img[-10:-1,:]
print(spec_img.shape)
cv2.imshow('Color image', spec_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#graph_spectrogram(wav_file_path, spec_img)