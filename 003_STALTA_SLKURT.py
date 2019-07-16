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

# SL KURT
d = data_tr

# Set data parameter
ls = 200
# ls = 300
ll = ls * 10

# Set Zero
data_STK = np.array(np.zeros(len(d)), float)
data_LTK = np.array(np.zeros(len(d)), float)

# STK
for j in range(len(d) - (ls - 1)):
    ds = d[j: ls + j]
    means = np.sum(ds) / ls
    sels = np.array(np.zeros(ls), float)
    for i in range(ls):
        sels[i] = (ds[i] - means) ** 2
    sumsels = np.sum(sels)
    vars = sumsels / (ls - 1)
    STK_sels = np.array(range(ls), float)
    for i in range(ls):
        STK_sels[i] = (ds[i] - means) ** 4
    STK_sumsels = np.sum(STK_sels)
    STK = STK_sumsels / ((ls - 1) * (vars ** 2))
    data_STK[int(j + np.ceil(ls / 2))] = STK

# LTK
for j in range(len(d) - (ll - 1)):
    dl = d[j:ll + j]
    mean1 = np.sum(dl) / ll
    sell = np.array(np.zeros(ll), float)
    for i in range(ll):
        sell[i] = (dl[i] - mean1) ** 2
    sumsels = np.sum(sell)
    var1 = sumsels / (ll - 1)
    LTK_sell = np.array(range(ll), float)
    for i in range(ll):
        LTK_sell[i] = (dl[i] - mean1) ** 4
    LTK_sumsell = np.sum(LTK_sell)
    LTK = LTK_sumsell / ((ll - 1) * (var1 ** 2))
    data_LTK[int(j + np.ceil(ll / 2))] = LTK

data_SLKurt = np.array(np.zeros(len(d)), float)
for i in range(len(d)):
    data_SLKurt[i] = data_STK[i] / (data_LTK[i] + (10 ** (-10)))

data_SLKurt_norm = np.array(np.zeros(len(d)), float)
data_SLKurt_mean = np.mean(data_SLKurt[int(np.ceil(ll / 2)): int(len(d) - np.ceil(ll / 2) + 1)])
selisih = np.array(np.zeros(len(d)), float)
for i in range(len(d)):
    selisih[i] = (data_SLKurt[i] - data_SLKurt_mean) ** 2
sumSTD = np.sum(selisih[int(np.ceil(ll / 2)): int(len(d) - np.ceil(ll / 2) + 1)])
data_SLKurt_STD = np.sqrt(sumSTD / (len(data_SLKurt[int(np.ceil(ll / 2)): int(len(d) - np.ceil(ll / 2) + 1)]) - 1))
data_SLKurt_norm[int(np.ceil(ll / 2)):int(len(d) - np.ceil(ll / 2) + 1)] = (data_SLKurt[
                                                                      int(np.ceil(ll / 2)):int(
                                                                          len(d) - np.ceil(ll / 2) + 1)] - \
                                                                      np.array(np.ones(len(data_SLKurt[
                                                                                     int(np.ceil(ll / 2)):int(
                                                                                         len(d) - np.ceil(
                                                                                             ll / 2) + 1)])),
                                                                            float) * data_SLKurt_mean) / \
                                                                     (np.array(len(data_SLKurt[
                                                                                int(np.ceil(ll / 2)):int(
                                                                                    len(d) - np.ceil(ll / 2) + 1)]),
                                                                            float) * data_SLKurt_STD)

# RECURSIVE STA LTA
RSL = recursive_sta_lta(data_tr, int(5 * 50), int(10 * 200))

max_RSL = RSL.max()
norm_RSL = np.zeros((len_data))
for i in range(len_data- 1):
    norm_RSL[i] = RSL[i] / max_RSL

# Correlation AIC & STA/LTA

new_correl = np.correlate(data_SLKurt_norm, norm_RSL, "same")
max_new_corr = new_correl.max()
Norm_corr = np.zeros((len_data))
for i in range(len_data - 1):
    Norm_corr[i] = new_correl[i] / max_new_corr

# Result to plotting
yy = plt.subplot(4, 1, 1)
plt.plot(x_axis, data_tr)
plt.title("SL KURT AND STA LTA")

plt.subplot(4, 1, 2,  sharex = yy)
plt.plot(x_axis, data_SLKurt_norm)

plt.subplot(4, 1, 3, sharex = yy)
plt.plot(x_axis, norm_RSL)

plt.subplot(4, 1, 4, sharex = yy)
plt.plot(x_axis, Norm_corr)
plt.show()

# print((norm_AIC))
print(len(norm_RSL))
print(len(new_correl))