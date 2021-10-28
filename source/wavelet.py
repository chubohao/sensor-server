import pywt
import matplotlib.pyplot as plt
import numpy as np
import librosa
import scipy
print(pywt.families(short=False))


wavename = 'cgau8'
fc = pywt.central_frequency(wavename)
print(fc)
cparam = 2 * fc * 64
scales = cparam / np.arange(64, 1, -1)

signal, sr = librosa.load("../data/rawdata/audio/20211028-202112-077218.wav", sr=16000, dtype=np.float64)
print(signal.shape)
time = np.arange(0, 0.5, 1.0 / 16000) # 16000
# 绘制波形
plt.figure(figsize=(8, 5))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
plt.subplot(311)
plt.title("Raw Data")
plt.plot(time, signal)

plt.subplot(312)
coeff, freq = pywt.cwt(signal, scales, wavename)
print(freq.shape)
plt.title("Wavelet Feature")
plt.contourf(time, freq, abs(coeff))

plt.subplot(313)
plt.title("SFFT Feature")
signal, sr = librosa.load("../data/rawdata/audio/20211028-202112-077218.wav", sr=16000, dtype=np.float64)
f, t, ps = scipy.signal.stft(signal, fs=16000, nperseg=128, noverlap=1, boundary=None, padded=None)
print(abs(ps).shape)
plt.imshow(abs(ps), origin="lower",aspect='auto')

plt.show()
