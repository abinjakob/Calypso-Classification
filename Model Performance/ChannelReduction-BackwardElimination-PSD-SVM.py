# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 15:20:56 2024

Channel Selection based on Backward Elimination with PSD and SVM 
-----------------------------------------------------------------
This script is used for the channel reduction for SSVEP experiment using SVM

Concept:
    step 01: kick-out one channel at a time and calculate model performance 
    step 02: find the channel with least affect on model performance. ie. the channel
             reduction did not affect the model performance much or improved the performance
    step 03: delete this channel from data 
    step 04: goto step 01 until preferred channel size is achieved

Feature used   : All PSD

Classification : SVM classifier with 5-Fold crossvalidation
                - spliting data using train_test_split
                - scaling using StandarScalar
                - hyperparameter tuning using GridSearchCV
    
@author: Abin Jacob
         Carl von Ossietzky University Oldenburg
         abin.jacob@uni-oldenburg.de
"""


#%% libraries

import mne
import numpy as np
import matplotlib.pyplot as plt
import os.path as op

from sklearn.metrics import confusion_matrix, accuracy_score, PrecisionRecallDisplay, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#%% parameters

# filter
l_freq = 0.1 
h_freq = None
# epoching 
tmin = -0.2
tmax = 4

# Events
event_id = {'stim_L15': 13, 'stim_L20': 14, 'stim_R15': 15, 'stim_R20': 16}
event_names = list(event_id.keys())
foi = [15, 20, 15, 20] # Freqs of interest

# PSD computation
fmin = 1.0
fmax = 100
# Show filter
show_filter = False

#%% load data 

rootpath = r'L:\Cloud\NeuroCFN\RESEARCH PROJECT\Research Project 02\Classification\Data'
# EEGLab file to load (.set)
filename = 'P02_SSVEP_preprocessed24Chans.set'
filepath = op.join(rootpath,filename)
# load file in mne 
raw = mne.io.read_raw_eeglab(filepath, eog= 'auto', preload= True)
a = raw.info

#Preprocess the data
# extracting events 
events, _ = mne.events_from_annotations(raw, verbose= False)
epochs = mne.Epochs(
    raw, 
    events= events, 
    event_id= [event_id['stim_L15'], event_id['stim_L20'], event_id['stim_R15'], event_id['stim_R20']], 
    tmin=tmin, tmax=tmax, 
    baseline= None, 
    preload= True, 
    reject={'eeg': 3.0}) # Reject epochs based on maximum peak-to-peak signal amplitude (PTP)
                         # No rejection due to very high value 

#%% Frequency analysis - Calculate power spectral density (PSD)

sfreq = epochs.info["sfreq"]
spectrum = epochs.compute_psd(
    method="welch",
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
psds, freqs = spectrum.get_data(return_freqs=True)

#%% creating the label vector

# Create a label vector
labels = epochs.events[:,2]
for i in range(0,len(labels)):
    if labels[i]==13 or labels[i]==15:
        labels[i] = 15
    else:
        labels[i] = 20

# Refine psds to frequency range around [12,45]
freq_range = range(np.where(np.floor(freqs) == 5)[0][0], np.where(np.ceil(freqs) == 45)[0][0])

#%% channel reduction - backward rejection method using PSD and SVM

# channel names
chans = raw.info.ch_names.copy()
# create a copy of psds for the channel reduction
psd_calc = psds
chanDeleted =[]
modelAcc = []

# loop over all channels 
for run in range(psds.shape[1]-1):
    
    # initalising vectors
    acc_temp = []
    precision_temp = []
    recall_temp = []
    f1score_temp = []
    # loop to run for all channels 
    for chan2delete in range(psd_calc.shape[1]):
        
        # delete channel temporarily
        psd_temp = np.delete(psd_calc, obj=chan2delete, axis=1)
        
        # choosing only selected range for freq power (12Hz - 25Hz)
        X = psd_temp[:,:,freq_range]
        # flatten the 3d EEG PSD matrix into vectors for classifier
        trials, chan, timepts = X.shape
        X = X.reshape((trials, -1))
        y = labels # labels
        
        # split the dataset into trainning and testing set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
        # define a pipeline with preprocessing (scaling) and SVM classifier
        pipeline = make_pipeline(StandardScaler(), SVC())
    
        # parameter grid for SVM
        param_grid = {
            'svc__C': [1],  # SVM regularization parameter
            'svc__gamma': [0.001],  # Kernel coefficient for 'rbf'
            'svc__kernel': ['sigmoid']  # Kernel type
        }
    
        # apply cros-validaion on training set to find best SVM parameters
        clf = GridSearchCV(pipeline, param_grid, cv=5)
        # train the pipeline
        clf.fit(X_train, y_train)
        # make predictions
        y_pred = clf.predict(X_test)
    
        # calculate model performance
        # accuracy
        acc_temp.append(accuracy_score(y_test, y_pred)) 
        # precision (positive predictive value)
        precision_temp.append(precision_score(y_test, y_pred, labels=[15,20], average= 'weighted')) 
        # recall (sensitivy or true positive rate)
        recall_temp.append(recall_score(y_test, y_pred, labels=[15,20], average= 'weighted')) 
        # f1 score (equillibrium between precision and recall)
        f1score_temp.append(f1_score(y_test, y_pred, labels=[15,20], average= 'weighted'))
    
    # calculating the performance for each channel    
    clf_performance = [(acc_temp[i] + precision_temp[i] + recall_temp[i] + f1score_temp[i]) for i in range(len(acc_temp))]   
    # finding the index of channel with least effect on the model performance 
    chanIdx = clf_performance.index(max(clf_performance))
    modelAcc.append(acc_temp[chanIdx])
    # deleting that channel from the 
    psd_calc = np.delete(psd_calc, obj=chanIdx, axis=1)
    psd_calc.shape[1]
    chanDeleted.append(chans[chanIdx])
    chans.pop(chanIdx)

plt.plot(np.arange(1, len(chanDeleted)+1), modelAcc, 'bs--', linewidth = 0.6)
plt.title('SVM Model Performance after Channel Reduction (kernel: sigmoid, c:1, gamma:0.001)')
plt.ylabel('Model Performance')
plt.xlabel('no of channels reduced')
plt.grid(color= 'lightgrey', alpha= 0.3)

# Annotate each point with its number
for i, (xi, yi) in enumerate(zip(np.arange(1, len(chanDeleted)+1), modelAcc)):
    plt.text(xi, yi, chanDeleted[i], color='blue', fontsize=12, ha='right', va='bottom')
