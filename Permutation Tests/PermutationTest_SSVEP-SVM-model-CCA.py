# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:15:39 2024

Permutaiotn Test of the SVM model for SSVEP
-------------------------------------------

A permutation test is carried to check the validity of the SVM model 
The model is run 1000 times with shuffled data and a histogram is plotted


Feature used   : CCA
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

#%% computing CCA 

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
# generating time vector
t = np.linspace(0, duration, tpts, endpoint= False)

# initialising array to store features
CCAfeatures = []

# loop over epochs 
for iEpoch in range(numEpochs):
    # extract the X array
    X_data = eegEpoch[iEpoch,:,:].T
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
        Y_data = np.column_stack((sine1, cos1, sine2, cos2))
        
        # performing CCA
        # considering the first canonical variables
        cca = CCA(n_components= 1)
        # compute cannonical variables
        cca.fit(X_data, Y_data)
        # return canonical variables
        Xc, Yc = cca.transform(X_data, Y_data)
        corr = np.corrcoef(Xc.T, Yc.T)[0,1]
        
        # store corr values for current epoch
        epochFeat.append(corr)
    
    # store features
    CCAfeatures.extend(epochFeat)

#%% Create feature and label vector

# create labels 
labels = epochs.events[:,2]
for i in range(0,len(labels)):
    if labels[i]==13 or labels[i]==15:
        labels[i] = 15
    else:
        labels[i] = 20

# feature vector (X)
X = np.array(CCAfeatures).reshape(numEpochs, -1)
# label vector (y)
y = labels 

#%% PERMUTATION TEST

# number of test runs 
testTrials = 1000

# initalising vectors
PTacc = []
PTprecision = []
PTrecall = []
PTf1score = []

# model performance observed (with real data) 
modelAcc = 1
modelPrecision = 1
modelRecall = 1
mdelF1score = 1

for iRun in range(testTrials):
    # shuffling labels 
    y = shuffle(y)

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
    # make predictions
    y_pred = clf.predict(X_test)    
    # generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # calculate model performance
    # accuracy
    PTacc.append(accuracy_score(y_test, y_pred))
    # precision (positive predictive value)
    precision = precision_score(y_test, y_pred, labels=[15,20], average= 'weighted')
    PTprecision.append(precision)
    # recall (sensitivy or true positive rate)
    recall = recall_score(y_test, y_pred, labels=[15,20], average= 'weighted')
    PTrecall.append(recall)
    # f1 score (equillibrium between precision and recall)
    f1score = f1_score(y_test, y_pred, labels=[15,20], average= 'weighted')
    PTf1score.append(f1score)
    
    # reporting after every 50 trials
    if iRun % 50 == 0:
        print(f'Running the permutation trial: {iRun}')
    
    # PrecisionRecallDisplay.from_estimator(clf, X_test, y_test)
    # PrecisionRecallDisplay.from_predictions(clf, y_test, y_pred)

#%% plot histogram

# calculating the emperical chance levels 
totalSamples = len(labels)
classCount = np.unique(labels, return_counts=True)[1]
chanceLevel = np.max(classCount) / totalSamples

plt.figure(figsize=(10,8))
# plot accuracy
plt.subplot(2,2,1)
plt.hist(PTacc, bins=40, color='skyblue', edgecolor='black')  # Adjust the number of bins as needed
# Add a vertical line for the observed accuracy
plt.axvline(x=modelAcc, color='red', linestyle='dashed', linewidth=1, label='model accuracy')
plt.axvline(x=chanceLevel, color='magenta', linestyle='dashed', linewidth=1, label='emp chance level')  
plt.xlim([0,1])  
plt.title('Accuracy')
plt.legend()

# plot precision
plt.subplot(2,2,2)
plt.hist(PTprecision, bins=40, color='skyblue', edgecolor='black')  # Adjust the number of bins as needed
# Add a vertical line for the observed accuracy
plt.axvline(x=modelPrecision, color='red', linestyle='dashed', linewidth=1, label='model precision')  
plt.axvline(x=chanceLevel, color='magenta', linestyle='dashed', linewidth=1, label='emp chance level')  
plt.xlim([0,1]) 
plt.title('Precision')
plt.legend()

# plot recall
plt.subplot(2,2,3)
plt.hist(PTrecall, bins=40, color='skyblue', edgecolor='black')  # Adjust the number of bins as needed
# Add a vertical line for the observed accuracy
plt.axvline(x=modelRecall, color='red', linestyle='dashed', linewidth=1, label='model recall')  
plt.axvline(x=chanceLevel, color='magenta', linestyle='dashed', linewidth=1, label='emp chance level')  
plt.xlim([0,1]) 
plt.title('Recall')
plt.legend()

# plot f1 score
plt.subplot(2,2,4)
plt.hist(PTf1score, bins=40, color='skyblue', edgecolor='black')  # Adjust the number of bins as needed
# Add a vertical line for the observed accuracy
plt.axvline(x=mdelF1score, color='red', linestyle='dashed', linewidth=1, label='model f1 score')  
plt.axvline(x=chanceLevel, color='magenta', linestyle='dashed', linewidth=1, label='emp chance level')  
plt.xlim([0,1]) 
plt.title('F1 score')
plt.legend()

plt.suptitle("Permutation Test of SVM Model with CCA features")
plt.show()