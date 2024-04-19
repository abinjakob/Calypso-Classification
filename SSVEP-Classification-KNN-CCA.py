# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:28:02 2024

Classification of the SSVEP signal using KNN 
---------------------------------------------
Feature used: CCA Correlation Values for stimulus frequencies and its harmonics

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
from sklearn.neighbors import KNeighborsClassifier
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
# event_id = {'stim_L15': 13, 'stim_L20': 14, 'stim_R15': 15, 'stim_R20': 16}
event_id = {'stim_L15': 10, 'stim_L20': 11, 'stim_R15': 12, 'stim_R20': 13}
event_names = list(event_id.keys())
foi = [15, 20, 15, 20] # Freqs of interest

#%% load data 

rootpath = r'L:\Cloud\NeuroCFN\RESEARCH PROJECT\Research Project 02\Classification\Data'
# EEGLab file to load (.set)
filename = 'P04_SSVEP_preprocessed.set'
filepath = op.join(rootpath,filename)
# load file in mne 
raw = mne.io.read_raw_eeglab(filepath, eog= 'auto', preload= True)
a = raw.info

# re-referencing 
# raw.set_eeg_reference(ref_channels=['E28'])

#Preprocess the data
# extracting events 
events, eventinfo = mne.events_from_annotations(raw, verbose= False)
epochs = mne.Epochs(
    raw, 
    events= events, 
    event_id= [event_id['stim_L15'], event_id['stim_L20'], event_id['stim_R15'], event_id['stim_R20']], 
    tmin=tmin, tmax=tmax, 
    baseline= None, 
    preload= True,
    event_repeated = 'merge',
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
    # if labels[i]==13 or labels[i]==15:
    if labels[i]==10 or labels[i]==12:
        labels[i] = 15
    else:
        labels[i] = 20

# feature vector (X)
X = np.array(CCAfeatures).reshape(numEpochs, -1)
# label vector (y)
y = labels 

#%% KNN classifier with 5 fold cross-validation 

# split the dataset into trainning and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# define a pipeline with preprocessing (scaling) and KNN classifier
pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier())

# parameter grid for KNN
param_grid = {
    'kneighborsclassifier__n_neighbors': [3, 5, 7, 10, 15],
    'kneighborsclassifier__weights': ['uniform', 'distance'],
    'kneighborsclassifier__metric': ['euclidean', 'manhattan', 'minkowski'],
    'kneighborsclassifier__algorithm': ['ball_tree', 'kd_tree', 'brute'],
    'kneighborsclassifier__p': [1, 2],
    'kneighborsclassifier__leaf_size': [20, 30, 40]
}

# apply cros-validaion on training set to find best KNN parameters
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
precision = precision_score(y_test, y_pred, labels=[15,20], average= 'weighted')
# recall (sensitivy or true positive rate)
recall = recall_score(y_test, y_pred, labels=[15,20], average= 'weighted')
# f1 score (equillibrium between precision and recall)
f1score = f1_score(y_test, y_pred, labels=[15,20], average= 'weighted')

# print model performance 
print('Confusion Matrix')
print(cm)
print('Model Performance Metrics')
print(f'Accuracy: {accuracy*100:.2f}%')
print(f'Precision: {precision*100:.2f}%')
print(f'Recall: {recall*100:.2f}%')
print(f'F1 Score: {f1score*100:.2f}%')


#%% plotting the data 

plt.scatter(X[:,0][y==15], X[:,1][y==15], label='label 15')
plt.scatter(X[:,0][y==20], X[:,1][y==20], label='label 20')
plt.title('Feature Space')
plt.xlabel('cca coeff for 15hz')
plt.ylabel('cca coeff for 20hz')
plt.legend()


# CCA as classifier (FUN :P)
# pred = []
# for i in range(X.shape[0]):
#     if X[i][0] > X[i][1]:
#         pred.append(15)
#     elif X[i][0] < X[i][1]:
#         pred.append(20)
# accuracy_CCAclf = accuracy_score(y, pred)
# print(f'Accuracy: {accuracy_CCAclf*100:.2f}%')

#%%

plt.figure()
plt.scatter(X_test[:,0][y_test==15], X_test[:,1][y_test==15], label='label 15')
plt.scatter(X_test[:,0][y_test==20], X_test[:,1][y_test==20], label='label 20')
plt.scatter(X_test[:,0][y_pred==15], X_test[:,1][y_pred==15], label='pred 15', marker= 'o', facecolors= 'none', edgecolors='blue', linewidth=1)
plt.scatter(X_test[:,0][y_pred==20], X_test[:,1][y_pred==20], label='pred 20', marker= 'o', facecolors= 'none', edgecolors='red', linewidth=1)
plt.xlabel('cca coeff for 15hz')
plt.ylabel('cca coeff for 20hz')
plt.title(f'SSVEP K-NN Prediction (Acc: {accuracy*100:.2f}%)')
plt.legend()

