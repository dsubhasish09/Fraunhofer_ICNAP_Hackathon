# -*- coding: utf-8 -*-
"""
Created on Sat May 21 22:09:59 2022

@author: devdu
"""
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file = open("processed_data.pkl","rb")
(X_train, Y_train, X_test, Y_test, X_val, Y_val, preprocessed_data) = pickle.load(file)
file.close()


pos = X_train[Y_train==1].squeeze()#[0].reshape((1,500))
pos_ft=np.zeros((500,))
for p in pos:
    fft=np.fft.fft(p)
    power = fft.real**2 + fft.imag**2
    pos_ft += power/np.sum(power)
pos_ft/=pos.shape[0]

plt.figure(dpi=300)
pos_freq = np.fft.fftfreq(500)
plt.plot(pos_freq,pos_ft,label = "positive")


neg = X_train[Y_train==0].squeeze()#[123].reshape((1,500))
neg_ft=np.zeros((500,))
for n in neg:
    fft=np.fft.fft(n)
    power = fft.real**2 + fft.imag**2
    neg_ft += power/sum(power)
neg_ft/=neg.shape[0]


neg_freq = np.fft.fftfreq(500)
plt.plot(neg_freq,neg_ft,label = "negative")
plt.legend()


