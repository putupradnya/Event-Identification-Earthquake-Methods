import numpy as np
from numpy import correlate
import matplotlib.pyplot as plt
from obspy import *
from obspy.signal.trigger import recursive_sta_lta


# Akaike Information Criteria (AIC) First Break Identification
file = read("D:/1999-03-04-mw71-celebes-sea.miniseed")
data_idx = file.pop(3)
data_tr = data_idx.data
t = data_idx.times()
len_data = len(data_tr)

# X Axis length
x_axis = np.arange(len_data)
len_x = len(x_axis)

# AIC Formula
AIC = np.zeros((len_data))

for i in range(0, len_data - 1):
    a = i * np.log(np.var(data_tr[0:i]))
    b = (len_data - i - 1) * (np.log(np.var(data_tr[i + 1: len_data])))
    AIC[i] = a + b

len_AIC = len(AIC)


# Differential AIC with time series
Diff_AIC = np.zeros((len_AIC))
for i in range(len_AIC - 1):
    Diff_AIC[i] = ((AIC[i + 1] - AIC[i])/(x_axis[i+1] - (x_axis[i])))**2

for i in range(len_AIC - 1):
    if Diff_AIC[i] == np.inf:
        Diff_AIC[i] = 0

new_AIC = np.nan_to_num(Diff_AIC)

max_diff_data = new_AIC.max()
norm_AIC = np.zeros((len_AIC))

for i in range(len_AIC - 1):
    norm_AIC[i] = Diff_AIC[i] / max_diff_data

new_Norm_AIC = np.nan_to_num(norm_AIC)

pick_AIC = []
for i in range(len_AIC - 1):
    if new_Norm_AIC[i] > 0.4:
        pick_AIC.append(i)


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

pick_MER = []
for i in range(len_AIC - 1):
    if norm_MER[i] > 0.4:
        pick_MER.append(i)

# CORRELATION OF THE DATA
corr_data = np.correlate(new_Norm_AIC, norm_MER, "same")
max_corr = corr_data.max()
norm_corr = np.zeros((len_data))
for i in range(len_data - 1):
    norm_corr[i] = corr_data[i] / max_corr

# Result to plotting
yy = plt.subplot(4, 1, 1)
plt.plot(x_axis, data_tr)
plt.title("AIC and MER")
for i in range(len(pick_AIC)):
    plt.plot(4, 1, 1)
    plt.plot(pick_AIC[i], 0, '|b', MarkerSize=1000, color = 'red')

plt.subplot(4, 1, 2,  sharex = yy)
plt.plot(x_axis, new_Norm_AIC)


plt.subplot(4, 1, 3, sharex = yy)
plt.plot(x_axis, norm_MER)
# for i in range(len(pick_MER)):
#     plt.subplot(4, 1, 3)
#     plt.plot(pick_MER[i],  0, '|b', MarkerSize=1000)

plt.subplot(4, 1, 4, sharex = yy)
plt.plot(x_axis, norm_corr)
plt.show()