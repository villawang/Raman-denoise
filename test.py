# %% codecell
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm, trange
import pandas as pd
import math
import pdb


# atplotlib inline
# %config InlineBackend.figure_format='retina'

# %%
band_num = 1000
band = np.arange(0.1, band_num, 1)
n_peaks = 40 #randi([5,50],1);  % Number of peaks to include in the spectrum
frq = []
FWHM = []
amp = []
for n in range(n_peaks):
    frq.append(np.random.randint(100, 900, 1))
    FWHM.append(np.random.randint(1, 500, 1))
    amp.append(10*np.random.uniform())





# %%
# make spectrum
num_spectra = 10000
raman_spectrum = []
CARS_spectrum = []

for m in trange(num_spectra):
    Xi_res = np.zeros([band_num], dtype=complex)
    for n in range(n_peaks):
        amplitude = 2.5 * amp[n] * (1+np.random.uniform())/1
        res   =  ( amplitude * frq[n]*frq[n] ) / (frq[n]*frq[n]-band*band-1j*band*FWHM[n])
        Xi_res = Xi_res + res
    raman_spectrum_temp = Xi_res.imag

    if np.isnan(raman_spectrum_temp).any():
        continue

    a       =  -2e-8*np.random.uniform() + 1e-7
    b       =  -2e-8*np.random.uniform() + 1e-7
    c       =  -2*np.random.uniform() + 5
    d       =  -10*np.random.uniform() + 50
    temp =   (a*np.power(band, 3) + b*band**2 + c*band + d)
    nrb =   temp.copy() #.*(randi([1,50],1)/max(temp));

    CARS_spectrum_temp = abs(nrb)**2 + 2*nrb*Xi_res.real + abs(Xi_res)**2

    if np.isnan(CARS_spectrum_temp).any():
        continue
    raman_spectrum.append(raman_spectrum_temp)
    CARS_spectrum.append(CARS_spectrum_temp)

raman_spectrum = np.stack(raman_spectrum)
CARS_spectrum = np.stack(CARS_spectrum)



print(np.isinf(raman_spectrum).any())
print(np.isinf(CARS_spectrum).any())
print(np.isnan(raman_spectrum).any())
print(np.isnan(CARS_spectrum).any())


pd.DataFrame(raman_spectrum).to_csv('./data/Raman_spectrums_train.csv')
pd.DataFrame(CARS_spectrum).to_csv('./data/CARS_spectrums_train.csv')
