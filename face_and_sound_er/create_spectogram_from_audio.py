import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np
import os
import wave
import pylab


def graph_spectrogram(wav_file):
    sound_info, frame_rate = get_wav_info(wav_file)
    pylab.figure(num=None, figsize=(19, 12))
    #pylab.subplot(111)
    pylab.title('spectrogram of %r' % wav_file)
    pylab.specgram(sound_info, Fs=frame_rate)
    pylab.savefig(wav_file.split('.')[0]+'.png')


def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'int16')
    print(sound_info.shape)
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate


if __name__() == '__main__':
    wav_files = ['/Users/aryaman/research/FER_datasets/video/ErinBrockavich_shot_2.wav',
                 '/Users/aryaman/research/FER_datasets/video/ErinBrockavich_shot_3.wav',
                 '/Users/aryaman/research/FER_datasets/video/ErinBrockavich_shot_4.wav',
                 '/Users/aryaman/research/FER_datasets/video/ErinBrockavich_shot_5.wav']
    for wav_file in wav_files:
        graph_spectrogram(wav_file)

# sample_rate, samples = wavfile.read()
# frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
# spectrogram = spectrogram*1000000
# print(spectrogram, type(spectrogram), spectrogram.shape)
# #spectrogram = (spectrogram/np.max(spectrogram))*255
# plt.imshow(spectrogram)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()