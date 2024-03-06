% % EEG DATA PRE-Processing MAIN SCRIPT 
% ------------------------------------
% The script is used to extract the ERD features from the artefact corrected EEG data 
% 
% Pre-processing steps:
% 1. Filtering (Low-pass and High-pass filters)
% 2. Rereference to CAR
% 3. Epoching data
% 4. Baseline Correction
% 5. Reject artefactual epochs
% 
% Feature extraction
% 1. Calculating ERD for epoched data using 'ERDfeature' function 
% 2. Save the ERD file [n_trial x n_chans x erd_values]
% 
% the pre-processed data is then stored to the given folder
% 
% Author: Abin Jacob
% Date  : 06/03/2024

%% load data 

% start fresh (:
clear; clc; close all;

% add EEGLab to matlab path
addpath('L:\Cloud\SW\eeglab2024.0');
% folder path 
MAINPATH = 'L:\Cloud\NeuroCFN\RESEARCH PROJECT\EEG Analysis Scripts';
DATAPATH = fullfile(MAINPATH, 'Data Analysis',filesep);
% file to load 
FILENAME = 'P01_MI_ICAcleaned.set';

% open EEGLab
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
% load ICA cleaned file 
EEG = pop_loadset(FILENAME, DATAPATH); 

% set current directory 
cd(MAINPATH);

%% parameters 

% pre-processing parameters 
% broad band filtering 
HP = 1;                         % high-pass cut-off
HP_order = 826;                 % high-pass filter order    
LP = 40;                        % low-pass cut-off
LP_order = 776;                 % low-pass filter order 
% event markers 
events = {'left_execution','right_execution','left_imagery','right_imagery'};
% epoch duration 
epoch_start = -5;            
epoch_end = 6;
% baseline correction
baseline = [epoch_start*EEG.srate 0]; 
% reject artefactual epochs 
PRUNE = 4;

% parameters for ERD calculation in mu band 
mu = [8 12];
binsize = 30;
base_start = -4;
base_end = -2;
fs = EEG.srate; 

%% pre-processing 

% load ICA-clean file
%EEG = pop_loadset('L:\Cloud\NeuroCFN\RESEARCH PROJECT\EEG Analysis Scripts\temp_rawdata\MI_pilot02_ICAcleaned.set');

% low-pass filter (broad band)
EEG = pop_firws(EEG, 'fcutoff', LP, 'ftype', 'lowpass', 'wtype', 'hamming', 'forder', LP_order);
% high-pass filter (broad band)
EEG = pop_firws(EEG, 'fcutoff', HP, 'ftype', 'highpass', 'wtype', 'hamming', 'forder', HP_order);
% re-referencing to CAR
EEG = pop_reref(EEG, [], 'refstate',0);

% removing unnecessary event marker
event_pos = 1;      % position counter for the events other than stim onset
event_idx = [];     % array to store the index of the event other than stim onset
% loop over events 
for idx = 1: length(EEG.event)
    if ~ strcmp(EEG.event(idx).type, events)
        event_idx(event_pos) = idx;
        event_pos = event_pos +1;
    end
end 
% remove events which are not stim onset from the data
EEG = pop_editeventvals(EEG, 'delete', event_idx);
EEG = eeg_checkset(EEG);

% epoching 
EEG = pop_epoch(EEG, events, [epoch_start epoch_end], 'newname', 'MI_pilot_epoched','epochinfo', 'yes');
% reject artefactual epochs 
% joint probability-based artifact rejection (joint prob. > PRUNE (SD))
EEG = pop_jointprob(EEG, 1, [1:EEG.nbchan], PRUNE, PRUNE, 0, 1, 0);
EEG = eeg_checkset(EEG);
% baseline correction
EEG = pop_rmbase(EEG, baseline);
EEG = eeg_checkset(EEG);

%% calculating ERD for each epoch in Mu Band 

% narrow band filtering (Mu Band)
muEEG = pop_firws(EEG, 'fcutoff', mu(2), 'ftype', 'lowpass', 'wtype', 'hamming', 'forder', LP_order);
muEEG = pop_firws(muEEG, 'fcutoff', mu(1), 'ftype', 'highpass', 'wtype', 'hamming', 'forder', HP_order);

% loop over trials 
for iTrial = 1:size(muEEG.data,3)
    trialData = muEEG.data(:,:,1);
    muERD(iTrial,:,:) = ERDfeature(trialData, binsize, base_start, base_end, fs); 
end 

%% creating label vector 

% initialize labels with zeros
labels = zeros(1, size(EEG.event, 2));
% creating labels 
for iTrial = 1:size(EEG.event, 2)
    % check for 15 Hz trials
    if strcmp(EEG.event(iTrial).type, 'left_execution')
        labels(iTrial) = 1;
    elseif strcmp(EEG.event(iTrial).type, 'right_execution')
        labels(iTrial) = 2;
    elseif strcmp(EEG.event(iTrial).type, 'left_imagery')
        labels(iTrial) = 3;
    elseif strcmp(EEG.event(iTrial).type, 'right_imagery')
        labels(iTrial) = 4;
    end
end

%% save ERD values and event labels

% save ERD values
erdfilePath = 'L:\Cloud\NeuroCFN\RESEARCH PROJECT\Research Project 02\Classification\Data\miERDvalues.mat';
save(erdfilePath, 'muERD')

% save event values 
labelfilePath = 'L:\Cloud\NeuroCFN\RESEARCH PROJECT\Research Project 02\Classification\Data\miEventvalues.mat';
save(labelfilePath, 'labels')



    


