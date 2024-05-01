# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:20:54 2024

Classification of the Spatial Auditory Attention using Template Matching
------------------------------------------------------------------------
Based on procedure mentioned in Bleichner et al 2016
DOI: 10.1088/1741-2560/13/6/066004


Feature used: Normalised Cross-Correlation Function (NCF) between each trial 
              and the templates.
                  
              - Templates are the average ERP of left attended trails 
              and right attended triasl
              
Classification: Leave-one-out Cross Validation Template Matching Approach

@author: Abin Jacob
         Carl von Ossietzky University Oldenburg
         abin.jacob@uni-oldenburg.de
         
"""

#%% libraries 

import mne
import numpy as np
import matplotlib.pyplot as plt
import os.path as op
from scipy.io import loadmat

from scipy.signal import correlate
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

#%% load data 

rootpath = r'/Users/abinjacob/Documents/02. NeuroCFN/Research Module/RM02/Data'
# EEGLab file to load (.set)
filename = 'P04_SAT_AllProcessed.set' 
filepath = op.join(rootpath,filename)
# load file in mne 
raw = mne.io.read_raw_eeglab(filepath, eog= 'auto', preload= True)

# eeg paramters 
sfreq = raw.info['sfreq']
# eeg signal
EEG = raw.get_data()
nchannels, nsamples = EEG.shape
# channel names 
chnames = raw.info['ch_names'] 

# re-referencing 
raw.set_eeg_reference(ref_channels=['E25', 'E25'])

# extracting events 
events, eventinfo = mne.events_from_annotations(raw, verbose= False)

# loading correct trials 
trialsfile = 'P04Sat_CorrTrials.mat' 
corrTrialsData = loadmat(op.join(rootpath, trialsfile))
# correct trials
corrTrials = [item[0] for item in corrTrialsData['event_name'][0]]


#%% epoching
tmin = -0.25            
tmax = 3
# extracting event ids of correct trials from eventinfo
event_id =[eventinfo[corrTrials[idx]] for idx in range(len(corrTrials))]

# epoching 
epochs = mne.Epochs(
    raw, 
    events= events, 
    event_id= event_id, 
    tmin=tmin, tmax=tmax, 
    baseline= (tmin, 0), 
    preload= True,
    event_repeated = 'merge',
    reject={'eeg': 4.0}) # Reject epochs based on maximum peak-to-peak signal amplitude (PTP)

# event id of left attended trials
trlsLeft = [event_id[idx] for idx, trial in enumerate(corrTrials) if 'left' in trial]
# event id of right attended trials
trlsRight = [event_id[idx] for idx, trial in enumerate(corrTrials) if 'right' in trial]

#%% creating template after splitting the data into train and test sets 

# data vector
data = epochs.get_data()

# labels 
labels = []
for trial in corrTrials:
    if 'left' in trial:
        labels.append(0)
    elif 'right' in trial:
        labels.append(1)

# converting labels to numpy array
labels = np.array(labels) 

#%% functions to compute Normalised Cross-Correlation and Template Matching 

# function to create template
def createTemplate(X, y):
    templateLeft = np.mean(X[y==0], axis=0)
    templateRight = np.mean(X[y==1], axis=0)
    return np.stack((templateLeft, templateRight))

# function to compute Normalised Cross-Correlation  
def computeNCC(signal, template, jitter, start, stop):
    # shape of template
    ntemp, nchan, _ = template.shape
    # convert jitter to samples
    jitter = int(jitter * sfreq / 1000)
    # convert start and stop period to samples
    start = int(start * sfreq / 1000)
    stop = int(stop * sfreq / 1000)
    # initialise array to store ncc
    ncc = np.zeros((nchan, ntemp))
    # loop over channels
    for ichan in range(nchan):
        s = signal[ichan, start:stop]
        # loop over left and right templates for the channel
        for itemp in range(ntemp):
            t = template[itemp, ichan, start:stop]
            # normalise signal and tempate
            snorm = (s - np.mean(s)) / (np.std(s) * len(s))
            tnorm = (t - np.mean(t)) / np.std(t)
            # computing cross correlation
            corr = correlate(snorm, tnorm, mode= 'full')
            lag = np.arange(-len(snorm)+1, len(snorm))
            # find the index of maximal correlation within the jitter range
            jittrange = (lag >= -jitter) & (lag <= jitter)
            # find max corr valiue within the period 
            corrMAX = np.max(corr[jittrange])
            ncc[ichan, itemp] = corrMAX
    return ncc 

# function to compute the difference between left and right template correlation
def computeDiff(ncc):
    nchan, _ = ncc.shape
    diff = np.zeros(nchan) 
    for ichan in range(nchan):
        diff[ichan] = ncc[ichan,0] - ncc[ichan,1]
    value = np.nansum(diff)
    return value

#%% leave one out template matching

# initialising leave-one-out cross validation 
loo = LeaveOneOut()

# duration of ERP to consider (400ms to 2800ms - to exclude the onset and offset responses) 
start = 400
stop = 2800 
# jitter in ms 
jitter = 50

acc = []

for trainid,testid in loo.split(data):
    # split data and labels into train and test set
    trainData, testData = data[trainid], data[testid]
    trainLabel, testLabel = labels[trainid], labels[testid]
    
    # creating template for left and right trials (cond x nchan x tpts)
    template = createTemplate(trainData, trainLabel)
    # create signal matrix (nchan x tpts)
    signal = testData[0,:,:]
    # computing NCC 
    ncc = computeNCC(signal, template, jitter, start, stop)
    # find the difference of left and right NCC across channels
    value = computeDiff(ncc)
    
    # classificatin 
    if value > 0:
        # classify as left
        decision = 0
    elif value < 0:
        # classify as right
        decision = 1
    
    # computing accuracy
    if decision == testLabel:
        # true 
        acc.append(1)
    else:
        # false 
        acc.append(0)    

# final accuracy in percentage 
accuracy = (np.sum(acc)/ data.shape[0]) * 100    
print(f'{accuracy:.2f}%')    

