#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 14:32:44 2023

@author: abinjacob
"""

#%% libraries 

import numpy as np
import matplotlib.pyplot as plt
import mne
import pyriemann
from sklearn.model_selection import cross_val_score

#%% load data 

# EEGLab file to load (.set)
filename = 'SSVEP_pilot2_rawdata.set'
# filename = 'P03_SSVEP_rawdata.set'
filepath = '/Users/abinjacob/Documents/02. NeuroCFN/Research Module/EEG Analysis Scripts/temp_rawdata' 
fullpath = f'{filepath}/{filename}'

# load file in mne 
raw = mne.io.read_raw_eeglab(fullpath, eog= 'auto', preload= True)

#%% parameters for data analysis

# filtering 
window = 'hamming'
# high pass filter 
HP = 0.1
HP_order = 16501
# low pass filter 
LP = 45
LP_order = 776

# epoching 
tmin = -0.2
tmax = 4

#%% pre-processing the data 

# applying low-pass filter 
raw.filter(l_freq= None, h_freq= LP, filter_length= LP_order, fir_window= window)
# applying high-pass filter 
raw.filter(l_freq= HP, h_freq= None, filter_length= HP_order, fir_window= window)

# set common average reference
raw.set_eeg_reference("average", projection=False, verbose=False)

# constructing epochs 
event_id = {'stim_L15': 13, 'stim_L20': 14, 'stim_R15': 15, 'stim_R20': 16}
# extracting events 
events, _ = mne.events_from_annotations(raw, verbose= False)
# epoching 
epochs = mne.Epochs(
    raw, 
    events= events, 
    event_id= [event_id['stim_L15'], event_id['stim_L20'], event_id['stim_R15'], event_id['stim_R20']], 
    tmin= tmin, tmax= tmax, 
    baseline= None, 
    preload= True, 
    reject={'eeg': 3.0})


#%% calculating PSD

# extracting 15 Hz epochs
selected_epoch = epochs['13','15']
fmin = 1.0
fmax = 90.0
sfreq = epochs.info["sfreq"]

spectrum = selected_epoch.compute_psd(
    "welch",
    n_fft=int(sfreq * (tmax - tmin)),
    n_overlap=0,
    n_per_seg=None,
    tmin=tmin,
    tmax=tmax,
    fmin=fmin,
    fmax=fmax,
    window="boxcar",
    verbose=False,
)
psds15, freqs = spectrum.get_data(return_freqs=True)

# extracting 20 Hz epochs
selected_epoch = epochs['14','16']
fmin = 1.0
fmax = 90.0
sfreq = epochs.info["sfreq"]

spectrum = selected_epoch.compute_psd(
    "welch",
    n_fft=int(sfreq * (tmax - tmin)),
    n_overlap=0,
    n_per_seg=None,
    tmin=tmin,
    tmax=tmax,
    fmin=fmin,
    fmax=fmax,
    window="boxcar",
    verbose=False,
)
psds20, freqs = spectrum.get_data(return_freqs=True)


#%% preparing the features and labels 

# creating the data
X = np.append(psds15, psds20, axis=0)

# creating the labels
label15 = np.full(psds15.shape[0],15)
label20 = np.full(psds20.shape[0],20) 
Y = np.append(label15, label20, axis=0)

#%% classification

# estimate covariance matrices
cov = pyriemann.estimation.Covariances().fit_transform(X)

# cross validation
mdm = pyriemann.classification.MDM()

accuracy = cross_val_score(mdm, cov, Y)

print(accuracy.mean())


#%% plotting psd
plt.figure()
freq_range = range(np.where(np.floor(freqs) == 1.0)[0][0], np.where(np.ceil(freqs) == fmax - 1)[0][0])
# psds_plot = 10 * np.log10(psds)
psds_mean = psds.mean(axis=(0, 1))[freq_range]
plt.plot(freqs[freq_range], psds_mean, color="b")
plt.xlim(5,45)


     
    


