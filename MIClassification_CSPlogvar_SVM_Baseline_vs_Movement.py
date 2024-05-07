# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:14:20 2024

Classification of the MI vs Baeline using SVM
----------------------------------------------

The script is used for the offline classification of the MI EEG data between 
right hand ME/MI vs the baseline period (no movement or imagined movement).  

Feature used: log-var of CSP spatial filtred data

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
import math

#%% load data 

rootpath = r'/Users/abinjacob/Documents/02. NeuroCFN/Research Module/RM02/Data'
# EEGLab file to load (.set)
filename = 'MAD_MI_Betaband.set'
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

# epoch duration 
tmin = -5            
tmax = 4

# events
event_id = {'S  1': 1}
# event_id = {'right_execution': 13}
event_names = list(event_id.keys())

# epoching 
epochs = mne.Epochs(
    raw, 
    events= events, 
    event_id= [event_id['S  1']], 
    # event_id= [event_id['right_execution']], 
    tmin=tmin, tmax=tmax, 
    baseline= (tmin, 0), 
    preload= True,
    event_repeated = 'merge',
    reject={'eeg': 4.0}) # Reject epochs based on maximum peak-to-peak signal amplitude (PTP)

#%% preparing data for classification

# for each epoch -5s to -1s will be the baseline (no movement/ imagined movement) 
# and from 0s to 4s will be the period for movement/imagined movement  

# baseline period 
tminBase = -5
tmaxBase = -1
# movement/ imagined movement 
tminImg = 0
tmaxImg = 4

# finding samples of baseline and movement period 
epochDuration = list(range(tmin, tmax+1))
idminBase = int((epochDuration.index(tminBase)) * sfreq)
idmaxBase = int((epochDuration.index(tmaxBase)) * sfreq)
idminImg = int((epochDuration.index(tminImg)) * sfreq)
idmaxImg = int((epochDuration.index(tmaxImg)) * sfreq)

# extracting baseline period EEG 
eegBase = epochs.get_data()[:,:,idminBase:idmaxBase]
labelBase = np.zeros(eegBase.shape[0], int)
eegImg = epochs.get_data()[:,:,idminImg:idmaxImg]
labelImg = np.ones(eegImg.shape[0], int)

# create feature vector (X)
X = np.concatenate((eegBase, eegImg), axis=0)
# create label vector (y)
y = np.concatenate((labelBase, labelImg))

#%% compute CSP on whole data and plot components

# compute CSP on train set (using MNE csp)
ncomps = X.shape[1]
cspALL = CSP(n_components=ncomps,reg=None, log=None, transform_into = 'csp_space', norm_trace=False)
cspALL.fit(X, y)
 
# calculating the number of cols and rows for subplot
ncols = int(math.ceil(np.sqrt(ncomps)))  
nrows = int(math.ceil(ncomps / ncols))
# setting figure title
figtitle = 'Motor Execution CSP Patterns'
# creating figure
fig, ax = plt.subplots(nrows,ncols)
fig.suptitle(figtitle, fontsize=16)    
ax = ax.flatten()
for icomp in range(ncomps):
    # csp patterns 
    patterns = cspALL.patterns_[icomp].reshape(epochs.info['nchan'],-1)
    # creating a mne structure 
    evoked = mne.EvokedArray(patterns, epochs.info)
    # plotting topoplot 
    evoked.plot_topomap(times=0, axes=ax[icomp], show=False, colorbar=False)
    ax[icomp].set_title(f'Comp {icomp + 1}', fontsize=10)
# setting empty axes to false
for i in range(ncomps, len(ax)):
    ax[i].set_visible(False)  


#%% SVM classifier with 5 fold cross-validation 

# split the dataset into trainning and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# compute CSP on train set (using MNE csp)
ncomps = 10
csp = CSP(n_components=ncomps,reg=None, log=None, transform_into = 'csp_space', norm_trace=False)
trainCSP = csp.fit_transform(X_train, y_train)
testCSP = csp.transform(X_test)

# using log-var of CSP weights as features
X_train = np.log(np.var(trainCSP, axis=2))
X_test = np.log(np.var(testCSP, axis=2))

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





