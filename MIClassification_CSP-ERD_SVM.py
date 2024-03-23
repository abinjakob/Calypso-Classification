#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 20:39:29 2024

Classification of the MI signal using SVM
------------------------------------------

This code is implemented based on the youtube tutorial video:
https://youtu.be/EAQcu6DLAS0

Feature used: ERD calculated on CSP filtred data
              This is now used as features for classification. 

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

from sklearn.metrics import confusion_matrix, accuracy_score, PrecisionRecallDisplay, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh
#%% load data 

rootpath = r'/Users/abinjacob/Documents/02. NeuroCFN/Research Module/RM02/Data'
# EEGLab file to load (.set)
filename = 'P01_MI_ICAcleaned.set'
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

#%% paramters for preprocessing 

# filtering 
# narrow band filtering in mu band (8-12 Hz)
# high-pass filter 
hp = 1
# low-pass filter  
lp = 40
# window type
window = 'hamming'

# epoching
tmin = -5            
tmax = 6

# conditions 
cl_lab = ['left', 'right']
cl1 = 'left'
cl2 = 'right'
# muBand 
muband = [8,12]

# csp parameters
# number of components
k = 3           
# regularisation
reg = 1e-8      

# erd calculation parameters
basestart = -2
baseend = -1
# time of interest (ERD period to consider for classification)
toi = [0,2]
binsize = 30

#%% preprocess the data

# applying low-pass filter 
raw.filter(l_freq= None, h_freq= lp, l_trans_bandwidth= 'auto', h_trans_bandwidth= 'auto', filter_length= 'auto', fir_window= window)
# applying high-pass filter 
raw.filter(l_freq= hp, h_freq= None, l_trans_bandwidth= 'auto', h_trans_bandwidth= 'auto', filter_length= 'auto', fir_window= window)

# set common average reference
raw.set_eeg_reference("average", projection=False, verbose=False)

# Events
event_id = {'left_execution': 7, 'left_imagery': 8, 'right_execution': 13, 'right_imagery': 14}
event_names = list(event_id.keys())

# epoching 
epochs = mne.Epochs(
    raw, 
    events= events, 
    event_id= [event_id['left_execution'], event_id['left_imagery'], event_id['right_execution'], event_id['right_imagery']], 
    tmin=tmin, tmax=tmax, 
    baseline= (tmin, 0), 
    preload= True, 
    reject={'eeg': 4.0}) # Reject epochs based on maximum peak-to-peak signal amplitude (PTP)


#%% functions for calculating CSP and ERD 

# computing CSP (using scipy.linalg)
def CSPcompute(X, y, k, reg):
    # data for class 1
    s1 = X[y==int(cond[0])]
    # data for class 2
    s2 = X[y==int(cond[1])]
    
    # compute covariance matrix for each class
    cov1 = np.mean([np.cov(s1[i]) for i in range(len(s1))], axis=0)
    cov2 = np.mean([np.cov(s2[i]) for i in range(len(s2))], axis=0)
    
    # Add regularization to covariance matrices
    cov1 = cov1 + reg * np.eye(cov1.shape[0])
    cov2 = cov2 + reg * np.eye(cov2.shape[0])
    # compute generalised eigenvalues and eigenvectors
    evals, evecs = eigh(cov1, cov1 + cov2)
    
    # sort eigenvectors based on eigenvalues
    eigidx = np.argsort(evals)
    evals = evals[eigidx]
    evecs = evecs[:, eigidx]
    
    csp = np.concatenate((evecs[:,:k], evecs[:,-k:]), axis=1)
    return csp

# apply CSP
def CSPfit(X, csp):
    # initialise list to store transformed data
    dataTransform = []
    # loop over trials
    for trialvals in X:
        trialTransform = np.dot(csp.T, trialvals)
        dataTransform.append(trialTransform)
    return np.array(dataTransform)

# function to calculate ERD
def ERDcompute(data, binsize, basestart, baseend, tmin, tmax):  
    # data shape
    ntrls, nchans, nsamp = data.shape 
    # window size
    window = nsamp//binsize
    # initialising array to store bin tine points
    tbins = []
    
    # creating bins 
    # loop over window
    for t in range(window):
        start = t*binsize
        # tbins[t] = list(range(start, start+binsize))
        tbins.append(list(range(start, start + binsize)))
        
    ntrls, nchans, nsamp = data.shape 
    # initialise matrix to store power vals
    powerVals = np.zeros((ntrls, nchans, window, binsize))
    
    # calculating ERD based on Pfurtscheller(1999)
    # squaring the amplitude to obtain power for each trial
    # loop over channels 
    for itrl in range(ntrls):
        # loop over channels
        for ichan in range(nchans):
            # loop over bins
            for ibin in range(window):
                powerVals[itrl, ichan, ibin, :] = data[itrl, ichan, tbins[ibin]]**2
    # average across time samples
    powerAvg = np.mean(powerVals, axis=3)
    
    # calculating baseline period
    # duration of the epoch 
    epochDuration = list(range(tmin, tmax+1)) 
    # finding bins that contain the baseline start and end 
    basestartBin = int((epochDuration.index(basestart)*sfreq)/binsize)
    baseendBin = int((epochDuration.index(baseend)*sfreq)/binsize)
    
    # calculating ERD 
    # initialise matrix to store ERD
    erd = np.zeros((ntrls, nchans, window))
    # loop over trials 
    for itrl in range(ntrls):
        # loop over channels
        for ichan in range(nchans):
            baselineAvg = np.mean(powerAvg[itrl,ichan,basestartBin:baseendBin])
            erd[itrl, ichan, :] = ((powerAvg[itrl,ichan,:]-baselineAvg)/baselineAvg)*100
    return erd 


def ERDfeatures(erd, binsize, toi, tmin, tmax):
    # erd shape
    ntrls, nchans, nsamp = erd.shape
    # duration of the epoch 
    epochDuration = list(range(tmin, tmax+1)) 
    # finding bins that contain the time of interest (toi)
    t1 = int((epochDuration.index(toi[0])*sfreq)/binsize)
    t2 = int((epochDuration.index(toi[1])*sfreq)/binsize)
    # initalise matrix to store ERD at toi
    erdFeat = np.zeros((ntrls, nchans, t2-t1))
    # loop over trials 
    for itrl in range(ntrls):
        # loop over channels
        for ichan in range(nchans):
            # extracting ERD at toi
            erdFeat[itrl, ichan, :] = erd[itrl, ichan, t1:t2]
    return erdFeat

#%% prepare the data for classification 

# choose only the muband
muEpochs = epochs.filter(l_freq= None, h_freq= muband[1], l_trans_bandwidth= 'auto', h_trans_bandwidth= 'auto', filter_length= 'auto', fir_window= window)
muEpochs = epochs.filter(l_freq= muband[0], h_freq= None, l_trans_bandwidth= 'auto', h_trans_bandwidth= 'auto', filter_length= 'auto', fir_window= window)
# Execution Condition (7 & 13)
# cond = ['7', '13']
# Imagery condition (8 & 14)
cond = ['8', '14']

# create feature vector (X)
X = muEpochs[cond].get_data()
# label vector (y)
y = muEpochs[cond].events[:,2] 

#%% SVM classifier with 5 fold cross-validation 

# split the dataset into trainning and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -- compute CSP
# compute CSP weights on train set
csp = CSPcompute(X_train, y_train, k, reg)
# apply CSP weights on train and test data
trainCSP = CSPfit(X_train, csp)
testCSP = CSPfit(X_test, csp)

# -- compute ERD on CSP filtered data
# compute ERD on train data 
trainERD = ERDcompute(trainCSP, binsize, basestart, baseend, tmin, tmax)
# extract ERD only at stim period
trainERDfeat = ERDfeatures(trainERD, binsize, toi, tmin, tmax)
# compute ERD on test data 
testERD = ERDcompute(testCSP, binsize, basestart, baseend, tmin, tmax)
# extract ERD only at stim period
testERDfeat = ERDfeatures(testERD, binsize, toi, tmin, tmax)

# prepare train and test set for classification
X_train = np.mean(trainERDfeat, axis=2)
X_test = np.mean(testERDfeat, axis=2)

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
precision = precision_score(y_test, y_pred, labels=[cond[0],cond[1]], average= 'weighted')
# recall (sensitivy or true positive rate)
recall = recall_score(y_test, y_pred, labels=[cond[0],cond[1]], average= 'weighted')
# f1 score (equillibrium between precision and recall)
f1score = f1_score(y_test, y_pred, labels=[cond[0],cond[1]], average= 'weighted')

# print model performance 
print('Confusion Matrix')
print(cm)
print('Model Performance Metrics')
print(f'Accuracy: {accuracy*100:.2f}%')
print(f'Precision: {precision*100:.2f}%')
print(f'Recall: {recall*100:.2f}%')
print(f'F1 Score: {f1score*100:.2f}%') 


