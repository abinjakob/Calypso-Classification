# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 12:14:46 2024

Classification of the Motor Imagery signal using SVM 
----------------------------------------------------
Feature used: Avg ERD during the stim period from left and right hand conditions
              
Classification: SVM classifier with 5-Fold crossvalidation
                - spliting data using train_test_split
                - scaling using StandarScalar
                - hyperparameter tuning using GridSearchCV

The pre-processing and ERD calculation is done in matlab and the erd values are 
saved as 'miERDvalues.mat' which is then loaded into python for classification.
The trial types are saved as 'miTrialTypes'. 

Files:
-----
ERD file     : 'miERDvalues.mat' file stored as 'muERD' (type= struct)
               contains ERD values (trials x nchans x erd value) 
             
               background info:
                 Epoch:-5s to 6s
                 baseline period for erd: -4s to -2s
                 ME/MI period: 0s to 4s
                 binsize used for erd calc: 30

Event file    : 'miTrialTypes.mat' file stored as 'labels' (type= struct)
                contains the labels for the trials  
                (left_execution: 1, right_execution: 2, left_imagery: 3, right_imagery: 4)    
                                             
Matlab Script : MI_EEG_DataProcess.m (for pre-processing and erd calculation)
                                                         
@author: Abin Jacob
         Carl von Ossietzky University Oldenburg
         abin.jacob@uni-oldenburg.de

"""

#%% libraries

import numpy as np
import matplotlib.pyplot as plt
import os.path as op
from scipy.io import loadmat

from sklearn.metrics import confusion_matrix, accuracy_score, PrecisionRecallDisplay, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#%% load ERD data 

# main file directory 
rootpath = r'L:\Cloud\NeuroCFN\RESEARCH PROJECT\Research Project 02\Classification\Data'

# erd file name 
ERD_filename = 'miERDvalues.mat'
# load erd file 
ERD_data = loadmat(op.join(rootpath,ERD_filename))
# store erd values 
muERD = ERD_data['muERD']

# labels file name 
label_filename = 'miTrialTypes.mat'
# load erd file 
label_data = loadmat(op.join(rootpath,label_filename))
# store erd values 
labels = label_data['labels'].flatten()

#%% extracting the average erd values

# eeg sampling rate 
srate = 500
# binsize used for erd calculation
binsize = 30
# epoch duration 
epoch_start = -5
epoch_end = 6
epoch_duration = list(range(epoch_start, epoch_end+1))
# event duration  
event_start = 0
event_end = 3
# bins with me/mi period erd
st = int(((epoch_duration.index(event_start)) * srate) / binsize)
en = int(((epoch_duration.index(event_end)) * srate) / binsize)

# extracting erd values during me/mi period 
erd = muERD[:,:,st:en]

# average erd values during me/mi period
erd_avg = np.mean(erd, axis=2)
# taking the values at C3(index=7) and C4(index=8) electrodes 
erd_vals = erd_avg[:, 7:9]


#%% preparing data for classification
# trial labels = execution: 1 & 2 ; imagery: 3 & 4

# trials for SVM classification
trials = [1,2]
# finding the indices of the specified trial conditions 
idx = np.where((labels == trials[0]) | (labels == trials[1]))

# preparing the feature vector (X)
X = erd_vals[idx]
# X = erd_avg[idx]
# preparing the label vector (y)
y = labels[idx]

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
precision = precision_score(y_test, y_pred, labels=[trials[0],trials[1]], average= 'weighted')
# recall (sensitivy or true positive rate)
recall = recall_score(y_test, y_pred, labels=[trials[0],trials[1]], average= 'weighted')
# f1 score (equillibrium between precision and recall)
f1score = f1_score(y_test, y_pred, labels=[trials[0],trials[1]], average= 'weighted')

# calculating the emperical chance levels 
totalSamples = len(y)
classCount = np.unique(y, return_counts=True)[1]
chanceLevel = np.max(classCount) / totalSamples

# print model performance 
print('Confusion Matrix')
print(cm)
print('Model Performance Metrics')
print(f'Accuracy: {accuracy*100:.2f}%')
print(f'Precision: {precision*100:.2f}%')
print(f'Recall: {recall*100:.2f}%')
print(f'F1 Score: {f1score*100:.2f}%')
print(f'Emp. Chance Level: {chanceLevel*100:.2f}%')

#%% plotting the data 

plt.scatter(X[:,0][y==trials[0]], X[:,1][y==trials[0]], label='left hand')
plt.scatter(X[:,0][y==trials[1]], X[:,1][y==trials[1]], label='right hand')
plt.title('Feature Space')
plt.xlabel('ERD at C3')
plt.ylabel('ERD at C4')
plt.legend()


# CCA as classifier (FUN :P)
pred = []
for i in range(X.shape[0]):
    if X[i][0] > X[i][1]:
        pred.append(15)
    elif X[i][0] < X[i][1]:
        pred.append(20)
accuracy_CCAclf = accuracy_score(y, pred)
print(f'Accuracy: {accuracy_CCAclf*100:.2f}%')

#%%

plt.figure()
plt.scatter(X_test[:,0][y_test==trials[0]], X_test[:,1][y_test==trials[0]], label='label 15')
plt.scatter(X_test[:,0][y_test==trials[1]], X_test[:,1][y_test==trials[1]], label='label 20')
plt.scatter(X_test[:,0][y_pred==trials[0]], X_test[:,1][y_pred==trials[0]], label='pred 15', marker= 'o', facecolors= 'none', edgecolors='blue', linewidth=1)
plt.scatter(X_test[:,0][y_pred==trials[1]], X_test[:,1][y_pred==trials[1]], label='pred 20', marker= 'o', facecolors= 'none', edgecolors='red', linewidth=1)
plt.xlabel('cca coeff for 15hz')
plt.ylabel('cca coeff for 20hz')
plt.title('SVM Linear Kernal Prediction')
plt.legend()
