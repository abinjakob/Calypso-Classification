#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 19:48:35 2024

Classification of the SSVEP signal using Riemannian Geometry 
------------------------------------------------------------

Feature used : PSD 
              
Classifier  : Riemannian Geometry based Classifier - Minimum Distance to Mean (RMDM)
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

from sklearn.metrics import confusion_matrix, accuracy_score, PrecisionRecallDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pyriemann.classification import MDM
from pyriemann.estimation import Covariances

from sklearn.utils import shuffle

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

rootpath = r'/Users/abinjacob/Documents/01. Calypso/EEG Coding/SSVEP/DATA'
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

#%% preparing data for classification

# Create a label vector
labels = epochs.events[:,2]

for i in range(0,len(labels)):
    if labels[i]==13 or labels[i]==15:
        labels[i] = 15
    else:
        labels[i] = 20
        
# Refine psds to frequency range around [12,25]
freq_range = range(np.where(np.floor(freqs) == 12)[0][0], np.where(np.ceil(freqs) == 25)[0][0])
# Mean over freq bins
X = psds[:,:,freq_range]
y = labels # labels


#%% RMDM classifier with 5 fold cross-validation 

# split the dataset into trainning and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# define a pipeline with estimating covariancec matrix and RMDM classifier
pipeline = make_pipeline(Covariances(), MDM())

# parameter grid for RMDM classifier 
param_grid = {
    'mdm__metric': ['riemann', 'logeuclid', 'euclid']
    }

# apply cros-validaion on training set to find best SVM parameters
clf = GridSearchCV(pipeline, param_grid, cv=5)
# train the pipeline
clf.fit(X_train, y_train)

# display best parameters found by GridSearchCV
print(f'Best Parameters Found: {clf.best_params_}')

# make predictions
y_pred = clf.predict(X_test)

# generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# calculate model performance
# accuracy
accuracy = accuracy_score(y_test, y_pred)
# precision (positive predictive value)
precision = (tp)/(tp + fp)
# recall (sensitivy or true positive rate)
recall = (tp)/(tp + fn)
# f1 score (equillibrium between precision and recall)
f1score = (2 * precision * recall) / (precision + recall)

# print model performance 
print('Confusion Matrix')
print(cm)
print('Model Performance Metrics')
print(f'Accuracy: {accuracy*100:.2f}%')
print(f'Precision: {precision*100:.2f}%')
print(f'Recall: {recall*100:.2f}%')
print(f'F1 Score: {f1score*100:.2f}%')

# PrecisionRecallDisplay.from_estimator(clf, X_test, y_test)
# PrecisionRecallDisplay.from_predictions(clf, y_test, y_pred)


























