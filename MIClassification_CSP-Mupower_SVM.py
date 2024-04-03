# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:39:40 2024

Classification of the MI signal using SVM
------------------------------------------

The script is used for the offline classification of the MI EEG data.  

Feature used: Mu band power of CSP spatial filtred data

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
from matplotlib import mlab

from sklearn.metrics import confusion_matrix, accuracy_score, PrecisionRecallDisplay, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh

from mne.decoding import CSP

#%% load data 

rootpath = r'L:\Cloud\NeuroCFN\RESEARCH PROJECT\Research Project 02\Classification\Data'
# EEGLab file to load (.set)
filename = 'P01_MI_AllProcessed.set'
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

#%% epoching
tmin = -0.5            
tmax = 4

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
    event_repeated = 'merge',
    reject={'eeg': 4.0}) # Reject epochs based on maximum peak-to-peak signal amplitude (PTP)

#%% functions for calculating CSP and PSD 

# CSP functions
# calculating CSP based on steps mentioned in Kabir et al. (2023) (https://doi.org/10.3390/math11081921)
# steps:
#   compute normalised spatial cov matrices for each class (covmat)
#   average normalized cov matrices across trials (covavg)
#   calculate composite cov matrix (covcomp)
#   perform eigenvalue decomposition on covcomp
#   calculate whitening transformation matrix (P)
#   find projection matrix W


# function to compute normalised spatial cov matrices for each class (covmat)
# cov = E*E'/trace(E*E') where E is the EEG signal of a particular trial 
# (chan x samples) from a particular class
def computeNormCov(E):
    cov = np.cov(E)
    covmat = cov / np.trace(cov)
    return covmat

# function to compute average normalized cov matrices across trials (covavg)
def avgCovmat(data):
    covavg = np.mean([computeNormCov(trial) for trial in data], axis=0)
    return covavg

# computing CSP
def computeCSP(X, y, cond):   
    # data for class 1
    Ec1 = X[y==int(cond[0])]
    # data for class 2
    Ec2 = X[y==int(cond[1])]    
    # average normalized cov matrices for each class
    covavg1 = avgCovmat(Ec1)
    covavg2 = avgCovmat(Ec2)    
    # composite cov matrix
    covcomp = covavg1 + covavg2
    
    # eigenvalue decomposition of composite cov matrix
    evals, evecs = eigh(covcomp)    
    # sort eigenvectors based on eigenvalues
    eigidx = np.argsort(evals)[::-1]
    evals = evals[eigidx]
    evecs = evecs[:, eigidx]
    
    # whitening transformation matrix
    P = np.dot(evecs, np.dot(np.diag(np.sqrt(1 / evals)), evecs.T))    
    # transform covariance matrices
    covwhite1 = np.dot(P.T, np.dot(covavg1, P))
    covwhite2 = np.dot(P.T, np.dot(covavg2, P))
    
    # solve the generalized eigenvalue problem on the transformed matrices
    _, B = eigh(covwhite1 - covwhite2)    
    # CSP projection matrix
    W = np.dot(P, B)

    return W

# applying CSP weights on data
def CSPfit(W, data):
    # initialise list to store transformed data
    dataTransform = []
    # loop over trials
    for trialvals in data:
        trialTransform = np.dot(W.T, trialvals)
        dataTransform.append(trialTransform)
    return np.array(dataTransform)

# PSD functions
# function to compute PSD
def PSDcompute(data):
    # data shape
    ntrls, nchans, nsamps = data.shape 
    # calculating nfft 
    nfft = 2**(np.ceil(np.log2(nsamps)).astype(int))
    # freq resolution 
    nfreqs = nfft // 2 + 1
    # empty matrix to store psd values
    trial_psd = np.zeros((ntrls,nchans, nfreqs))
    # loop over trials 
    for itrl in range(ntrls):
        # loop over channels
        for ichan in range(nchans):
            # calculate PSD 
            (PSD, freqs) = mlab.psd(data[itrl, ichan,:], NFFT= nfft, Fs=sfreq)
            trial_psd[itrl, ichan, :] = PSD.ravel()
    return trial_psd, freqs

#%% prepare the data for classification 
# Execution Condition (7 & 13)
# cond = ['7', '13']
# Imagery condition (8 & 14)
cond = ['8', '14']

# create feature vector (X)
X = epochs[cond].get_data()
# label vector (y)
y = epochs[cond].events[:,2] 

#%% SVM classifier with 5 fold cross-validation 

# split the dataset into trainning and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # -- compute CSP using my own script
# # compute CSP weights on train set
# W = computeCSP(X_train, y_train, cond)
# # applying CSP weights on train and test set
# trainCSP = CSPfit(W, X_train)
# testCSP = CSPfit(W, X_test)

# -- compute CSP using mne script
csp = CSP(n_components=23, reg=None, log=None, transform_into = 'csp_space', norm_trace=False)
trainCSP = csp.fit_transform(X_train, y_train)
testCSP = csp.transform(X_test)

# -- compute PSD after CSP 
trainPSD, freqs = PSDcompute(trainCSP)
testPSD, freqs = PSDcompute(testCSP)
# finding index of mu freqs
muidx  = np.where((freqs>= 8) & (freqs<= 12))[0]

# using bandpower of mu band as features
X_train = np.mean(trainPSD[:,:,muidx], axis=2)
X_test = np.mean(testPSD[:,:,muidx], axis=2)

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

