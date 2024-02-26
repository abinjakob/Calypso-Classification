# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:56:54 2024

Evaluating the performance of SVM and RMDM Models 
-------------------------------------------------

Feature used:    
    1. All PSD
    2. PSD at Stim freqs
    3. PSD around Stim freqs (fb=1 Hz)
    4. PSD around Stim freqs (fb=.5 Hz)
    5. PSD around Stim freqs (fb=.1 Hz)
    6. PSD at Stim freqs and Seconds Harmonics 
    7. PSD around Stim freqs and Seconds Harmonics (fb=1 Hz)
    8. PSD around Stim freqs and Seconds Harmonics (fb=.5 Hz)
    9. PSD around Stim freqs and Seconds Harmonics (fb=.1 Hz)

Classifier:     
    1.SVM classifier with 5-Fold crossvalidation
                - Linear 
                - Polynomial
                - Sigmoid 
                - RBF
    2. Riemannian - Minimum Distance to Mean Classifier 
    
The scrip will plot a line graph for for each classifier with each features fed

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
from sklearn.covariance import LedoitWolf
from sklearn.base import BaseEstimator, TransformerMixin


from pyriemann.classification import MDM
from pyriemann.estimation import Covariances

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
filename = 'P02_SSVEP_preprocessed5Chans.set'
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

#%% preparing features for svm (2D)

# Create a label vector
labels = epochs.events[:,2]
for i in range(0,len(labels)):
    if labels[i]==13 or labels[i]==15:
        labels[i] = 15
    else:
        labels[i] = 20

# initialising dictionary to store features and labels
feature_X = {}
label_y = {}


# 1. ALL PSD -------
# Refine psds to frequency range around [12,25]
freq_range = range(np.where(np.floor(freqs) == 12)[0][0], np.where(np.ceil(freqs) == 25)[0][0])
# Mean over freq bins
X = psds[:,:,freq_range]
# flatten the 3d EEG PSD matrix into vectors for classifier
trials, chans, timepts = X.shape
feature_X[0] = X.reshape((trials, -1))
label_y[0] = labels # labels


# 2. PSD at Stim freqs -------
# initialising a dictionary to store the values
stimPSD = {}
# stimulation frequencies and second harmonics
stimF = [15, 20]
# loop over stimF
for iFreq in stimF:
    # find index of stim freqs
    freqIdx = np.where(freqs == iFreq)[0]                                        
    # psd values at stim freqs
    stimPSD[iFreq] = np.mean(psds[:,:, freqIdx], axis=-1)
# # create X vector 
feature_X[1] = np.concatenate((stimPSD[15], stimPSD[20]), axis=0)
# # create y vector
label_y[1] = np.concatenate([labels, labels])


# 3,4,5. PSD around Stim freqs -------
# initialising a dictionary to store the values
stimPSD = {}
# stimulation frequencies and second harmonics
stimF = [15, 20]
fb = [1, 0.5, 0.1]
# loop over fb
for idx, val in enumerate(fb):
    # loop over stimF
    for iFreq in stimF:
        # find index of stim freqs
        freqIdx = np.where((freqs >= iFreq-(val/2)) & (freqs <= iFreq+(val/2)))[0]                                        
        # psd values at stim freqs
        stimPSD[iFreq] = np.mean(psds[:,:, freqIdx], axis=-1)
    # # create X vector 
    feature_X[idx+2] = np.concatenate((stimPSD[15], stimPSD[20]), axis=0)
    # # create y vector
    label_y[idx+2] = np.concatenate([labels, labels])


# 6. PSD at Stim freqs and Seconds Harmonics -------
# initialising a dictionary to store the values
stimPSD = {}
# stimulation frequencies and second harmonics
stimF = [15, 20, 30, 40]
# loop over stimF
for iFreq in stimF:
    # find index of stim freqs
    freqIdx = np.where(freqs == iFreq)[0]                                        
    # psd values at stim freqs
    stimPSD[iFreq] = np.mean(psds[:,:, freqIdx], axis=-1)
# # create X vector 
feature_X[5] = np.concatenate((stimPSD[15], stimPSD[20], stimPSD[30], stimPSD[40]), axis=0)
# # create y vector
label_y[5] = np.concatenate([labels, labels, labels, labels])


# 7,8,9. PSD around Stim freqs and Seconds Harmonics -------
# initialising a dictionary to store the values
stimPSD = {}
# stimulation frequencies and second harmonics
stimF = [15, 20, 30, 40]
fb = [1, 0.5, 0.1]
# loop over fb
for idx, val in enumerate(fb):
    # loop over stimF
    for iFreq in stimF:
        # find index of stim freqs
        freqIdx = np.where((freqs >= iFreq-(val/2)) & (freqs <= iFreq+(val/2)))[0]                                        
        # psd values at stim freqs
        stimPSD[iFreq] = np.mean(psds[:,:, freqIdx], axis=-1)
    # # create X vector 
    feature_X[idx+6] = np.concatenate((stimPSD[15], stimPSD[20], stimPSD[30], stimPSD[40]), axis=0)
    # # create y vector
    label_y[idx+6] = np.concatenate([labels, labels, labels, labels])
    
    
#%% preparing features for RMDM (2D)

class CovarianceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cov_estimator = LedoitWolf()
    
    def fit(self, X, y=None):
        # Fit is not doing anything here as LedoitWolf does not require fitting in the traditional sense
        return self
    
    def transform(self, X):
        # Assuming X is a list of matrices (trials) for which we want to compute the covariance
        transformed = np.array([self.cov_estimator.fit(x).covariance_ for x in X])
        return transformed

# Create a label vector
labels = epochs.events[:,2]
for i in range(0,len(labels)):
    if labels[i]==13 or labels[i]==15:
        labels[i] = 15
    else:
        labels[i] = 20

# initialising dictionary to store features and labels
RMDMfeature_X = {}
RMDMlabel_y = {}


# 1. ALL PSD -------
# Refine psds to frequency range around [12,25]
freq_range = range(np.where(np.floor(freqs) == 12)[0][0], np.where(np.ceil(freqs) == 25)[0][0])
# Mean over freq bins
RMDMfeature_X[0] = psds[:,:,freq_range]
RMDMlabel_y[0] = labels # labels


# 2. PSD at Stim freqs -------
# initialising a dictionary to store the values
stimPSD = {}
# stimulation frequencies and second harmonics
stimF = [15, 20]
# loop over stimF
for iFreq in stimF:
    # find index of stim freqs
    freqIdx = np.where(freqs == iFreq)[0]                                        
    # psd values at stim freqs
    stimPSD[iFreq] = psds[:,:, freqIdx]
# # create X vector 
RMDMfeature_X[1] = np.concatenate((stimPSD[15], stimPSD[20]), axis=0)
# # create y vector
RMDMlabel_y[1] = np.concatenate([labels, labels])


# 3,4,5. PSD around Stim freqs -------
# initialising a dictionary to store the values
stimPSD = {}
# stimulation frequencies and second harmonics
stimF = [15, 20]
fb = [1, 0.5, 0.1]
# loop over fb
for idx, val in enumerate(fb):
    # loop over stimF
    for iFreq in stimF:
        # find index of stim freqs
        freqIdx = np.where((freqs >= iFreq-(val/2)) & (freqs <= iFreq+(val/2)))[0]                                        
        # psd values at stim freqs
        stimPSD[iFreq] = psds[:,:, freqIdx]
    # # create X vector 
    RMDMfeature_X[idx+2] = np.concatenate((stimPSD[15], stimPSD[20]), axis=0)
    # # create y vector
    RMDMlabel_y[idx+2] = np.concatenate([labels, labels])


# 6. PSD at Stim freqs and Seconds Harmonics -------
# initialising a dictionary to store the values
stimPSD = {}
# stimulation frequencies and second harmonics
stimF = [15, 20, 30, 40]
# loop over stimF
for iFreq in stimF:
    # find index of stim freqs
    freqIdx = np.where(freqs == iFreq)[0]                                        
    # psd values at stim freqs
    stimPSD[iFreq] = psds[:,:, freqIdx]
# # create X vector 
RMDMfeature_X[5] = np.concatenate((stimPSD[15], stimPSD[20], stimPSD[30], stimPSD[40]), axis=0)
# # create y vector
RMDMlabel_y[5] = np.concatenate([labels, labels, labels, labels])


# 7,8,9. PSD around Stim freqs and Seconds Harmonics -------
# initialising a dictionary to store the values
stimPSD = {}
# stimulation frequencies and second harmonics
stimF = [15, 20, 30, 40]
fb = [1, 0.5, 0.1]
# loop over fb
for idx, val in enumerate(fb):
    # loop over stimF
    for iFreq in stimF:
        # find index of stim freqs
        freqIdx = np.where((freqs >= iFreq-(val/2)) & (freqs <= iFreq+(val/2)))[0]                                        
        # psd values at stim freqs
        stimPSD[iFreq] = psds[:,:, freqIdx]
    # # create X vector 
    RMDMfeature_X[idx+6] = np.concatenate((stimPSD[15], stimPSD[20], stimPSD[30], stimPSD[40]), axis=0)
    # # create y vector
    RMDMlabel_y[idx+6] = np.concatenate([labels, labels, labels, labels])

#%% classification using SVM and RMDM 

kernal_types = ['poly', 'sigmoid', 'rbf', 'linear']
acc = [[] for _ in range(len(feature_X))]
# loop over features 
for iFeat in range(len(feature_X)):   
    # feature vector
    X = feature_X[iFeat]
    # label vector 
    y = label_y[iFeat]
    # split the dataset into trainning and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # SVM classifier 
    # define a pipeline with preprocessing (scaling) and SVM classifier
    pipeline_svm = make_pipeline(StandardScaler(), SVC())
    
    # loop over SVM kernals
    for kern in kernal_types:
        # parameter grid for SVM
        param_grid = {
            'svc__C': [0.1, 1, 10, 100],  # SVM regularization parameter
            'svc__gamma': [0.001, 0.01, 0.1, 1],  # Kernel coefficient for 'rbf'
            'svc__kernel': [kern]  # Kernel type
        }
        
        # apply cros-validaion on training set to find best SVM parameters
        clf_svm = GridSearchCV(pipeline_svm, param_grid, cv=5)
        # train the pipeline
        clf_svm.fit(X_train, y_train)
        # make predictions
        y_predsvm = clf_svm.predict(X_test)
        # save accuracies
        acc[iFeat].append(accuracy_score(y_test, y_predsvm))
    
    # feature vector for RMDM
    X = RMDMfeature_X[iFeat]
    # label vector RMDM
    y = RMDMlabel_y[iFeat]
    # split the dataset into trainning and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    if iFeat==0:
        # define a pipeline with estimating covariancec matrix and RMDM classifier
        pipeline_rmdm = make_pipeline(Covariances(), MDM())
    else:
        pipeline_rmdm = make_pipeline(CovarianceTransformer(), MDM())
            
    # parameter grid for RMDM classifier 
    param_grid_rmdm = {
        'mdm__metric': ['riemann']
        }

    # apply cros-validaion on training set to find best SVM parameters
    clf_rmdm = GridSearchCV(pipeline_rmdm, param_grid_rmdm, cv=5)
    # train the pipeline
    clf_rmdm.fit(X_train, y_train)

    # make predictions
    y_pred_rmdm = clf_rmdm.predict(X_test)
    # save accuracies
    acc[iFeat].append(accuracy_score(y_test, y_pred_rmdm))

#%% plotting results 

models = ['SVM-poly', 'SVM-sigmoid', 'SVM-rbf', 'SVM-linear', 'RMDM']
feature_vec = ['All PSD','PSD at Stim freq','PSD around Stim freqs (fb=1 Hz)', ' PSD around Stim freqs (fb=.5 Hz)', 'PSD around Stim freqs (fb=.1 Hz)',
               'PSD at at Stim freqs and Seconds Harmonics', 'PSD around Stim freqs and Seconds Harmonics (fb=1 Hz)', 'PSD around Stim freqs and Seconds Harmonics (fb=.5 Hz)',
               'PSD around Stim freqs and Seconds Harmonics (fb=.1 Hz)']
    
# loop over features 
for iFeat in range(len(feature_X)): 
    plt.plot(models, acc[iFeat], 'o--', linewidth = 0.6, label=f'{feature_vec[iFeat]}')
    
plt.ylabel('Accuracy')
plt.ylim([0.3,1])
plt.title('Model Performance Across Different Features')
plt.xticks(rotation=45)    
plt.legend(loc='upper left', bbox_to_anchor=(0, 1), prop={'size': 8})
plt.tight_layout()  
plt.show()




