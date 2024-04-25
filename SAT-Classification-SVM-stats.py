# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 16:42:01 2024

Classification of the Spatial Auditory Attention using SVM
----------------------------------------------------------
Feature used: Statistical features of ERP

Classification: SVM classifier with 5-Fold crossvalidation
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
from scipy.io import loadmat
from scipy.stats import skew, kurtosis

from sklearn.metrics import confusion_matrix, accuracy_score, PrecisionRecallDisplay, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#%% load data 

rootpath = r'L:\Cloud\NeuroCFN\RESEARCH PROJECT\Research Project 02\Classification\Data'
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

#%% feature extraction 

# channels to select 
chan2sel = [30, 6, 41, 1, 36, 2, 31, 0, 35]
# chan2sel = [1, 4, 5, 6, 7, 8]
# extract eeg data from selected channels 
eegdata = np.mean(epochs.get_data()[:,chan2sel,126:],axis= 1)
# number of tones in left stream
lefttones = 4
# number of tones in right stream
righttones = 5
# vector with left tone onsets
lefttpts = np.linspace(0,3,lefttones+1)
# vector with right tone onsets
rightttpts = np.linspace(0,3,righttones+1)
# index of left and right tone onsets except the first tone 
toneidx = (np.hstack((lefttpts[1:-1] * sfreq +1, rightttpts[1:-1] * sfreq +1))).astype(int)

# time duration analysed for each tone (150ms to 300ms post tone onset) 
st = .15
ed = .3 
tid = [int(st * sfreq), int(ed * sfreq)]

ntrls, _ = eegdata.shape
ERPfeatures = []
# loop over trials 
for itrl in range(ntrls):
    # loop over time frame
    feat = []
    for t in toneidx:
        # extracting data for current time points
        data = eegdata[itrl, t+tid[0]:t+tid[1]]
        
        # -- computing features
        mean = np.mean(data)                                     # mean
        stdv = np.std(data)                                      # standard deviation
        median = np.median(data)                                 # median 
        skewness = skew(data)                                    # skewness
        kurt = kurtosis(data)                                    # kurtosis
        waveform = np.sum(np.abs(np.diff(data)))                 # waveform length
        slopesign =  np.sum(np.diff(np.sign(np.diff(data))))     # slope sign change
        energy = np.sum(data ** 2)                               # energy    
        
        # store features within each trial
        feat.extend([mean, stdv, median, skewness, kurt, waveform, slopesign, energy])
    # store feature for each trial
    ERPfeatures.append(np.array(feat))

#%% create feature and label vector

# create labels 
labels = []
for trial in corrTrials:
    if 'left' in trial:
        labels.append(0)
    elif 'right' in trial:
        labels.append(1)

# feature vector (X)
X = np.array(ERPfeatures)
# label vector (y)
y = np.array(labels) 


#%% SVM classifier with 5 fold cross-validation 

# split the dataset into trainning and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# define a pipeline with preprocessing (scaling) and SVM classifier
pipeline = make_pipeline(StandardScaler(), SVC())

# parameter grid for SVM
param_grid = {
    'svc__C': [0.1, 1, 10, 100],  # SVM regularization parameter
    'svc__gamma': [0.001, 0.01, 0.1, 1],  # Kernel coefficient for 'rbf'
    'svc__kernel': ['linear', 'rbf', 'poly', 'sigmoid']  # Kernel type
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
precision = precision_score(y_test, y_pred, labels=[0,1], average= 'weighted')
# recall (sensitivy or true positive rate)
recall = recall_score(y_test, y_pred, labels=[0,1], average= 'weighted')
# f1 score (equillibrium between precision and recall)
f1score = f1_score(y_test, y_pred, labels=[0,1], average= 'weighted')

# print model performance 
print('Confusion Matrix')
print(cm)
print('Model Performance Metrics')
print(f'Accuracy: {accuracy*100:.2f}%')
print(f'Precision: {precision*100:.2f}%')
print(f'Recall: {recall*100:.2f}%')
print(f'F1 Score: {f1score*100:.2f}%')


#%% plotting results

stats = ['Mean', 'SD', 'Median', 'Skewness', 'Kurtosis', 'Waveform Length', 'Slope Sign Change', 'Energy'] 
tones = ['tone 2', 'tone 3', 'tone 4', 'tone 5']
streams = ['Left Tones', 'Right Tones']

# Create grouped bar plot
bar_width = 0.25
x = np.arange(len(tones))


# loop over stats 
for istat in range(len(stats)):
    fig, axs = plt.subplots(2)
    fig.suptitle(f'{stats[istat]}')
    for icond in range(2):
        ct = istat 
        for itone in range(len(tones)):            
            
            if itone == 3:
                axs[icond].bar(itone + bar_width, np.mean(X[:,ct+(3*len(stats))][y==icond]), bar_width, color = 'mediumslateblue', label='Left Tone')
            else:
                axs[icond].bar(itone, np.mean(X[:,ct][y==icond]), bar_width, label='Left Tone', color = 'darkslateblue')
                axs[icond].bar(itone + bar_width, np.mean(X[:,ct+(3*len(stats))][y==icond]), bar_width, color = 'mediumslateblue', label='Right Tone')
            ct = ct + len(stats) 
        axs[icond].set_xticks(x + bar_width / 2)
        axs[icond].set_xticklabels(tones)
        axs[icond].set_ylabel(f'{stats[istat]}')
        # axs[icond].legend()
        if icond == 0:
            titlestring = 'Left Attended Condition'
        else:
            titlestring = 'Right Attended Condition'              
        axs[icond].set_title(titlestring)
            
#%% averaged 

stats = ['Mean', 'SD', 'Median', 'Skewness', 'Kurtosis', 'Waveform Length', 'Slope Sign Change', 'Energy'] 
bar_width = 0.1
# loop over stats 
for istat in range(len(stats)):
    fig, axs = plt.subplots(2)
    fig.suptitle(f'{stats[istat]}')
    for icond in range(2):
        ct = istat 
        leftst = []
        rightst = []        
        for itone in range(len(tones)-1): 
            leftst.append(np.mean(X[:,ct][y==icond]))
            rightst.append(np.mean(X[:,ct+(3*len(stats))][y==icond]))
            ct = ct + len(stats)
        axs[icond].bar(icond, np.mean(leftst), bar_width, label='Left Tone', color = 'darkslateblue')
        axs[icond].bar(icond + bar_width, np.mean(rightst), bar_width, label='Right Tone', color = 'mediumslateblue')
        axs[icond].set_ylabel(f'{stats[istat]}')
        
#%% plotting feature vector 
comb = [19, 35]
plt.scatter(X[:,comb[0]][y==0], X[:,comb[1]][y==0], color = 'blue')
plt.scatter(X[:,comb[0]][y==1], X[:,comb[1]][y==1], color = 'orange')