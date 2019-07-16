import numpy as np
from numpy import correlate
import matplotlib.pyplot as plt
from obspy import *
from obspy.signal.trigger import recursive_sta_lta


# Akaike Information Criteria (AIC) First Break Identification
file = read("D:/Package_1526693365600-BMKG_815333_BMKG.mseed")
data_idx = file.pop(0)
data_tr = data_idx.data
t = data_idx.times()
len_data = len(data_tr)

# X Axis length
x_axis = np.arange(len_data)
len_x = len(x_axis)

# RECURSIVE STA LTA
RSL = recursive_sta_lta(data_tr, int(5 * 50), int(10 * 200))

max_RSL = RSL.max()
norm_RSL = np.zeros((len_data))
for i in range(len_data - 1):
    norm_RSL[i] = RSL[i] / max_RSL


# MODIFIED ENERGY RATIO

x = data_tr
print(len(x))

Af= np.fft.fft(x)
fmax = 1/(t[1]-t[0])
f = np.linspace(0,fmax,len(t))
Aff = np.real(Af[0:int(np.ceil(len(Af)/2))])
ff = f[0:int(np.ceil(len(Af)/2))]

idx = np.argsort(Aff)
ffi = ff[idx[len(idx)-1]]
for i in range(len(idx)-1):
    if(ffi == 0):
        ffi = ff[idx[len(idx) - (i+1)]]
    else:
        break

T = 1/ffi

print(T)

L = 15
print(L)
er = np.zeros(len(x))
mer = np.zeros(len(x))
for i in range(len(x)-int(2*L)):
    k = i + int(L)
    era = np.zeros(int(L))
    erb = np.zeros(int(L))
    for j in range(int(L)):
        era[j] = (pow(x[k-j],2))
        erb[j] = (pow(x[k+j],2))
    eras = np.sum(era)
    erbs = np.sum(erb)
    er[k] = eras/(erbs+10**(-6))
    mer[k] = (er[k]*np.abs(x[k]))**3

max_MER = mer.max()
norm_MER = np.zeros((len_x))
for i in range(len_data - 1):
    norm_MER[i] = mer[i] / max_MER

# CORRELATION OF THE DATA
corr_data = np.correlate(norm_RSL, norm_MER, "same")
max_corr = corr_data.max()
norm_corr = np.zeros((len_data))
for i in range(len_data - 1):
    norm_corr[i] = corr_data[i] / max_corr

# Result to plotting
yy = plt.subplot(4, 1, 1)
plt.plot(x_axis, data_tr)
plt.title("STA/LTA and MER")

plt.subplot(4, 1, 2,  sharex = yy)
plt.plot(x_axis, norm_RSL)

plt.subplot(4, 1, 3, sharex = yy)
plt.plot(x_axis, norm_MER)

plt.subplot(4, 1, 4, sharex = yy)
plt.plot(x_axis, norm_corr)
plt.show()