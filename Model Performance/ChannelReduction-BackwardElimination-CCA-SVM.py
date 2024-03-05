# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:21:40 2024

Channel Selection based on Backward Elimination with CCA and SVM 
-----------------------------------------------------------------
This script is used for the channel reduction for SSVEP experiment using SVM

Concept:
    step 01: kick-out one channel at a time and calculate model performance 
    step 02: find the channel with least affect on model performance. ie. the channel
             reduction did not affect the model performance much or improved the performance
    step 03: delete this channel from data 
    step 04: goto step 01 until preferred channel size is achieved

Feature used   : CCA Features

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

from sklearn.cross_decomposition import CCA

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

#%% channel reduction - backward rejection method using CCA and SVM

# create label vector for SVM classification
labels = epochs.events[:,2]
for i in range(0,len(labels)):
    if labels[i]==13 or labels[i]==15:
        labels[i] = 15
    else:
        labels[i] = 20

# parameters for CCA
# number of epochs and samples 
numEpochs, _, tpts = epochs.get_data().shape
# eeg data from the epocs 
eegEpoch = epochs.get_data()
# stimulation frequencies
freqs = [15, 20]
# sampling frequency
fs = epochs.info["sfreq"]
# duration of epochs 
duration = tpts/fs
# generating time vector for sin and cos signals 
t = np.linspace(0, duration, tpts, endpoint= False)

# channel names
chans = raw.info.ch_names.copy()
# create a copy of psds for the channel reduction
eegCalc = eegEpoch
chanDeleted =[]
modelAcc = []

# loop over all channels 
for run in range(eegEpoch.shape[1]-1):
    print(f'\nModel currenly in loop {run+1}.................')       # print status messgage
    
    # initalising arrays to store model performance 
    acc_temp = []
    precision_temp = []
    recall_temp = []
    f1score_temp = []
    
    # initialising array to store features
    CCAfeatures = []
    
    # loop to kick each channel out in the current run
    for chan2delete in range(eegCalc.shape[1]):
        print(f'currenly in channel {chan2delete+1}.................')       # print status messgage
        # delete channel temporarily
        eegTemp = np.delete(eegCalc, obj=chan2delete, axis=1)
        
        # computing CCA ---      
        # loop over epochs 
        for iEpoch in range(numEpochs):
            # extract the X array
            ccaX_data = eegTemp[iEpoch,:,:].T
            # initialise array to store featues for each epoch
            epochFeat = []
            # loop over frequencies
            for i, iFreq in enumerate(freqs):    
                # create the sine and cosine signals for 1st harmonics
                sine1 = np.sin(2 * np.pi * iFreq * t)
                cos1 = np.cos(2 * np.pi * iFreq * t)
                # create the sine and cosine signals for 2nd harmonics
                sine2 = np.sin(2 * np.pi * (2 * iFreq) * t)
                cos2 = np.cos(2 * np.pi * (2 * iFreq) * t)
                
                # create Y vector 
                ccaY_data = np.column_stack((sine1, cos1, sine2, cos2))
                
                # performing CCA
                # considering the first canonical variables
                cca = CCA(n_components= 1)
                # compute cannonical variables
                cca.fit(ccaX_data, ccaY_data)
                # return canonical variables
                Xc, Yc = cca.transform(ccaX_data, ccaY_data)
                corr = np.corrcoef(Xc.T, Yc.T)[0,1]
                
                # store corr values for current epoch
                epochFeat.append(corr)
                # end of loop over freqs 
            
            # store features
            CCAfeatures.extend(epochFeat)
            # end of loop over epochs 
        
        # SVM classification ---
        # feature vector (X)
        X = np.array(CCAfeatures).reshape(numEpochs, -1)
        # label vector (y)
        y = labels 
        
        # split the dataset into trainning and testing set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
        # define a pipeline with preprocessing (scaling) and SVM classifier
        pipeline = make_pipeline(StandardScaler(), SVC())
    
        # parameter grid for SVM
        param_grid = {
            'svc__C': [1],  # SVM regularization parameter
            'svc__gamma': [1],  # Kernel coefficient for 'rbf'
            'svc__kernel': ['poly']  # Kernel type
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
        # end of loop kick each channel out in the current run 
    
    # calculating the performance for each channel    
    clf_performance = [(acc_temp[i] + precision_temp[i] + recall_temp[i] + f1score_temp[i]) for i in range(len(acc_temp))]   
    # finding the index of channel with least effect on the model performance 
    chanIdx = clf_performance.index(max(clf_performance))
    modelAcc.append(acc_temp[chanIdx])
    # deleting that channel from the 
    eegCalc = np.delete(eegCalc, obj=chanIdx, axis=1)
    eegCalc.shape[1]
    chanDeleted.append(chans[chanIdx])
    chans.pop(chanIdx)
    # end of loop over all channels 

# plot the model performance 
plt.plot(np.arange(1, len(chanDeleted)+1), modelAcc, 'bs--', linewidth = 0.6)
plt.title('SVM Model Performance after Channel Reduction (feat: CCA, kernel: poly, c:1, gamma:1)')
plt.ylabel('Model Performance')
plt.xlabel('no of channels reduced')
plt.grid(color= 'lightgrey', alpha= 0.3)

# Annotate each point with its number
for i, (xi, yi) in enumerate(zip(np.arange(1, len(chanDeleted)+1), modelAcc)):
    plt.text(xi, yi, chanDeleted[i], color='blue', fontsize=12, ha='right', va='bottom')
