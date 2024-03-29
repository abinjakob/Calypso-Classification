# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 17:48:24 2024

Classification of the SSVEP signal using SVM 
---------------------------------------------
Feature used: PSD around the stimulation frequencies
              [The mean power around the 15Hz and 20Hz were extracted from all 
               4 conditions which is then used as the features for classification]
              
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

from sklearn.metrics import confusion_matrix, accuracy_score, PrecisionRecallDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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

# # Plot PSD - all 4 conditions
# fig, axs = plt.subplots(2, 2, sharex="all", sharey="none", figsize=(8, 5))
# axs = axs.reshape(-1)
# freq_range = range(np.where(np.floor(freqs) == fmin)[0][0], np.where(np.ceil(freqs) == fmax - 1)[0][0])
# for i in range(0,4):  
#     # Get all events for specific event ID (condition)
#     condition = event_names[i]
#     idx = epochs.events[:,2]==event_id[condition]
#     # Select only corresponding epochs
#     psds_plot = 10 * np.log10(psds[idx,:,:])
#     psds_mean = psds.mean(axis=(0, 1))[freq_range] # mean over trials and channels
#     psds_std = psds.std(axis=(0, 1))[freq_range]
    
#     # Plot
#     axs[i].plot(freqs[freq_range], psds_mean, color="b")
#     axs[i].fill_between(
#         freqs[freq_range], psds_mean - psds_std, psds_mean + psds_std, color="b", alpha=0.2
#         )
#     axs[i].set(title=f"PSD spectrum {condition}", ylabel="Power Spectral Density [dB]")
#     axs[i].set_xlim(5,45)

# fig.show()

#%% extracting only power around stimulation frequencies 

# initialising a dictionary to store the values
stimPSD = {}
# stimulation frequencies and second harmonics
stimF = [15, 20]
# stimF = [15, 20, 30, 40]
# frequency band to be considered around stim freqs
fb = .5

for iFreq in stimF:
    # find index of stim freqs
    # freqIdx = np.where((freqs >= iFreq-(fb/2)) & (freqs <= iFreq+(fb/2)))[0]   # band of frequency around stim freqs
    freqIdx = np.where(freqs == iFreq)[0]                                        # exactly at stim freqs
    # psd values at stim freqs
    stimPSD[iFreq] = np.mean(psds[:,:, freqIdx], axis=-1)


#%% preparing data for classification

# Create a label vector
labels = epochs.events[:,2]
for i in range(0,len(labels)):
    if labels[i]==13 or labels[i]==15:
        labels[i] = 15
    else:
        labels[i] = 20
        
# create X vector 
X = np.concatenate((stimPSD[15], stimPSD[20]), axis=0)
# create y vector
y = np.concatenate([labels, labels])

# # create X vector
# X = np.concatenate((stimPSD[15], stimPSD[20], stimPSD[30], stimPSD[40]), axis=0)
# # create y vector
# y = np.concatenate([labels, labels, labels, labels])


#%% sanity check by shuffling the features and labels
# X = shuffle(X)        ### comment this line
# y = shuffle(y)        ### comment this line

# # Plot some samples or features
# plt.figure(figsize=(10, 5))
# for i in range(5):  # Plot first 5 samples
#     plt.subplot(2, 5, i + 1)
#     plt.plot(X[i])
#     plt.title(f'Sample {i+1}')
#     plt.xlabel('Feature Index')
#     plt.ylabel('Feature Value')
# plt.tight_layout()
# plt.show()

# # plot feature in scatter plot
# # Extracting labels 15 and 20 from X
# X_label_15 = X[y == 15]
# X_label_20 = X[y == 20]

# # # Plotting
# plt.scatter(X_label_15[:,0:600], X_label_15[:,600:1200], label='Label 15')
# plt.scatter(X_label_20[:,0:600], X_label_20[:,600:1200], label='Label 20')

# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('Scatter plot of features colored by labels')
# plt.legend()
# plt.show()

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
precision = (tp)/(tp + fp)
# recall (sensitivy or true positive rate)
recall = (tp)/(tp + fn)
# f1 score (equillibrium between precision and recall)
f1score = (2 * precision * recall) / (precision + recall)

# print model performance 
print('Confusion Matrix')
print(cm)
print('Model Performance Metrics')
print(f'Accuracy: {accuracy*100:.2f}%')
print(f'Precision: {precision*100:.2f}%')
print(f'Recall: {recall*100:.2f}%')
print(f'F1 Score: {f1score*100:.2f}%')

# PrecisionRecallDisplay.from_estimator(clf, X_test, y_test)
# PrecisionRecallDisplay.from_predictions(clf, y_test, y_pred)



