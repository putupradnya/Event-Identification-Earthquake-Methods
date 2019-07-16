import numpy as np
from numpy import correlate
import matplotlib.pyplot as plt
from obspy import *
from obspy.signal.trigger import recursive_sta_lta


# Akaike Information Criteria (AIC) First Break Identification
file = read("D:/1999-03-04-mw71-celebes-sea.miniseed")
data_idx = file.pop(3)
data_tr = data_idx.data
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
for i in range(10, len_AIC - 10):
    if (new_Norm_AIC[i]  > 0.3):
        pick_AIC.append(i)

print(pick_AIC)
print(len(pick_AIC))

# Recursive STA LTA
RSL = recursive_sta_lta(data_tr, int(5 * 50), int(10 * 200))

max_RSL = RSL.max()
norm_RSL = np.zeros((len_AIC))
for i in range(len_AIC - 1):
    norm_RSL[i] = RSL[i] / max_RSL

#Correlation AIC & STA/LTA

new_correl = np.correlate(new_Norm_AIC, norm_RSL, "same")
max_new_corr = new_correl.max()
Norm_corr = np.zeros((len_AIC))
for i in range(len_AIC - 1):
    Norm_corr[i] = new_correl[i] / max_new_corr

# PLOTING
yy = plt.subplot(4, 1, 1)
plt.plot(x_axis, data_tr)
plt.title("Akaike Information Criteria FB Identification")
for i in range(len(pick_AIC)):
    plt.plot(4, 1, 1)
    plt.plot(pick_AIC[i], 0, '|b', MarkerSize=1000, color = 'red')

plt.subplot(4, 1, 2,  sharex = yy)
plt.plot(x_axis, new_Norm_AIC)


plt.subplot(4, 1, 3, sharex = yy)
plt.plot(x_axis, norm_RSL)

plt.subplot(4, 1, 4, sharex = yy)
plt.plot(x_axis, Norm_corr)
plt.show()

print((norm_AIC))
print(len(norm_RSL))
print(len(new_correl))
