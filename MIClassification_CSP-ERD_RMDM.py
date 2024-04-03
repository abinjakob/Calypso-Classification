# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 20:39:29 2024

Classification of the MI signal using RMDM
------------------------------------------

The script is used for the offline classification of the MI EEG data.  

Feature used: ERD calculated from CSP spatial filtred data

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
from matplotlib import mlab

from sklearn.metrics import confusion_matrix, accuracy_score, PrecisionRecallDisplay, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh

from mne.decoding import CSP

from pyriemann.classification import MDM
from pyriemann.estimation import Covariances

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

#%% paramters for preprocessing 

# epoching
tmin = -3            
tmax = 4  

# erd calculation parameters
basestart = -2
baseend = -1
# time of interest (ERD period to consider for classification)
toi = [0,2]
binsize = 30

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


#%% functions for calculating CSP and ERD

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


# ERD functions
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
X_train = trainERDfeat
X_test = testERDfeat

# define a pipeline with estimating covariancec matrix and RMDM classifier
pipeline = make_pipeline(Covariances(), MDM())  # if value error occours - add Tikhonov regularisation: Covariances(reg=1e-5)

# parameter grid for RMDM classifier 
param_grid = {
    'mdm__metric': ['riemann']
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
    
