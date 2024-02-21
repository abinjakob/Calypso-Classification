# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 19:32:08 2024

Classification of the SSVEP signal using SVM 
---------------------------------------------
Feature used: GED filtred data and calculated PSD for Left trials 
              This is now used as features for classification. 
              
EEG preprocessing, signal processing, GED weighting, creating feature vector and 
label vectors was done using MatLab code. 

Files:
-----
Matlab File      : SSVEP_GED.m
Feature file (X) : gedPSD_Features_X.mat (shape (n_samples x n_features): 44 x 81993)
                   stored in 'features_X' 
Labels file (y)  : gedPSD_Labels_y.mat (shape: 1 X 44)
                   stored in 'labels_y' 

@author: Abin Jacob
         Carl von Ossietzky University Oldenburg
         abin.jacob@uni-oldenburg.de
         
"""

#%% libraries

import numpy as np
import matplotlib.pyplot as plt
import os.path as op
from scipy.io import loadmat

from sklearn.metrics import confusion_matrix, accuracy_score, PrecisionRecallDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.utils import shuffle

#%% load features (X) and labels (y) matlab files 

# main file directory 
rootpath = r'L:\Cloud\NeuroCFN\RESEARCH PROJECT\Research Project 02\Classification\Data'

# load features (X)
file_X = 'gedPSD_Features_X.mat'
filepath_X = op.join(rootpath,file_X)
data_X = loadmat(filepath_X)
# store features
X = data_X['features_X']

# load labels (y)
file_y = 'gedPSD_Labels_y.mat'
filepath_y = op.join(rootpath,file_y)
data_y = loadmat(filepath_y)
# store features
y = data_y['labels_y'].flatten()

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

# plot feature in scatter plot
# Extracting labels 15 and 20 from X
# X_label_15 = X[y == 15]
# X_label_20 = X[y == 20]

# # Plotting
# plt.scatter(X_label_15[:,0:4000], X_label_15[:,4000:8000], label='Label 15')
# plt.scatter(X_label_20[:,0:4000], X_label_20[:,4000:8000], label='Label 20')

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
    # 'svc__kernel': ['rbf']  # Kernel type
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

# calculating the emperical chance levels 
totalSamples = len(y)
classCount = np.unique(y, return_counts=True)[1]
chanceLevel = np.max(classCount) / totalSamples

# print emperical chance level
print(f'Chance level  : {chanceLevel*100:.2f}%')

