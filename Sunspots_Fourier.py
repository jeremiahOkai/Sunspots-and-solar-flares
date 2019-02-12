import numpy as np
import matplotlib.pyplot as plt

spotsdata = np.loadtxt('sunspot_monthly.csv', delimiter=';') # Loading the sunspots data
spotmean = spotsdata[:,3]

maxval = max(spotmean)
spotmean_norm = spotmean / maxval # Normalise the values to lie between 0 and 1

t = np.arange(len(spotmean_norm))

fourier_sunspots = np.fft.fft(spotmean_norm)
fourier_sunspots = fourier_sunspots[1:500]
freq = np.fft.fftfreq(t.shape[-1])
freq = freq[1:500]

max_freq = freq[np.argmax(fourier_sunspots)]

print("Peak frequency = {}\nPeak period = {}".format(max_freq, 1 / max_freq))

plt.plot(freq.real, fourier_sunspots.real, 'b-')
plt.xlim(0, 0.02)
plt.xlabel('Frequency (months^-1)')
plt.ylabel('Transformed Sunspot Number')
plt.savefig('FFT Sunspots.png')
plt.clf()