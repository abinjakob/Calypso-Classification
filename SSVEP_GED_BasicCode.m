% % SSVEP signal processing with GED
% ------------------------------------
% The code performs the SSVEP signal processing and apply generalised
% eigen decomposition (GED) as described in Cohen(2022)
% 
% Pre-processing steps:
% 1. Filtering (Low-pass and High-pass filters)
% 2. Rereference to CAR
% 3. Epoching data
% 4. Baseline Correction
% 5. Reject artefactual epochs
%
% Data-procesing steps:
% 6. Sub-epoching the data
% 7. Calculating GED weights 
% 8. Calculating PSD for sub-epoched data
% 9. Plot PSD for selected channel and condition
% 10. Feature vecor and label vector si created for classification 
%
% the pre-processed data is then stored to the given folder
% 
% Author: Abin Jacob
% Date  : 05/02/2024

%% pre-processing  
% clear; clc; close all;

% add EEGLab to matlab path
addpath('L:\Cloud\SW\eeglab2023.1');

% parameters for pre-proecssing
% filtering 
% high-pass filter 
HP = 0.1;                       % cut-off
HP_order = 826;                 % filter order    
% low-pass filter  
LP = 45;                        % cut-off
LP_order = 776;                 % filter order 

% epoching
% event markers 
events = {'stim_L20','stim_L15','stim_R20','stim_R15'};
epoch_start = -0.2;            
epoch_end = 4;   

% baseline correction
% defining baseline for baseline correcion
baseline = [epoch_start*EEG.srate 0];   
% reject artefactual epochs 
PRUNE = 4;

% low-pass filter
EEG = pop_firws(EEG, 'fcutoff', LP, 'ftype', 'lowpass', 'wtype', 'hamming', 'forder', LP_order);
% high-pass filter
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

%% weighting channels using GED 

freqs = {20, 15, 20, 15};

for i = 1:length(events)
    % selecting the event data (subepoching)
    EEG_temp = pop_selectevent(EEG, 'type', events{i},'renametype', events{i}, 'deleteevents', 'off', 'deleteepochs', 'on', 'invertepochs', 'off');  
    
    % preparing the signal matrix (matS)
    % banpass filtering data around the stimulus frequency
    narrowFilt = pop_eegfiltnew(EEG_temp, freqs{i}, freqs{i}, [], 0, [], 0);
    matS = narrowFilt.data;
    % reshaping data 
    matS = reshape(matS, EEG_temp.nbchan, []);
    % making the data mean centred
    matS = bsxfun(@minus, matS, mean(matS,2));
    % calculate the covariance matrix for S
    covmatS = (matS * matS') / (size(matS,2) - 1);       % (dividing by a normalisation factor of n-1)
    
    % preparing the reference matrix (matR)
    % broadband filtered data 
    matR = EEG_temp.data;
    % reshaping data 
    matR = reshape(matR, EEG_temp.nbchan, []);
    % making the data mean centred
    matR = bsxfun(@minus, matR, mean(matR,2));
    % calculate covariance matrix for R
    covmatR = (matR * matR') / (size(matR,2) - 1);       % (dividing by a normalisation factor of n-1)
    
    % performing GED
    [evecs, evals] = eig(covmatS, covmatR);
    % sorting diagonal values of eigenvalues in ascending
    [~, sidx] = sort(real(diag(evals)));
    % sorting eigenvectors based on sorted eigenvalues
    evecs = real(evecs(:, sidx));
    % storing the sorted eigenvectors
    gedWeights(i).evecs = evecs;
    
    % weighting the channel data based on GED
    gedWeights(i).data = reshape( (matR'*evecs(:,end))',EEG_temp.pnts,EEG_temp.trials);
    
    % calculating PSD
    % parameters for pwelch
    gedData = gedWeights(i).data;
    window_length = size(gedData,1);
    overlap = window_length / 2;

    % calculating power 
    % loop over trials 
    for iTrial = 1:size(gedData,2)
        % computing psd usign pwelch
        [pxx, f] = pwelch(gedData(:,iTrial), hamming(window_length), overlap, 2^nextpow2(window_length*4), EEG_temp.srate);
        % computing psd usign pwelch
        pxx_all(:,iTrial) = pxx;
    end
    gedWeights(i).psd = pxx_all;
    gedWeights(i).f = f;
    
    % plotting PSD 
    figure;
    plot(f, mean(pxx_all,2));
    set(gca, 'xlim', [5 45]);
    stims = {'Left 20Hz', 'Left 15Hz', 'Right 20Hz', 'Right 15Hz'};
    title(['PSD of GED weighted data for ' stims{i} ' condition'])
    xlabel('Frequency (Hz)');
    ylabel('Power (norm.)');
    
    % plotting topoplots
    figure;
    % calculating pseudo-inverse to get the mapping from component space back to electrode space
    tmpmap = pinv(evecs');
    % plotting the spatial distribution of the last component 
    topoplot(tmpmap(:,end)./max(abs(tmpmap(:,end))), EEG.chanlocs);
    title(['Topoplot of GED weighted data for ' stims{i} ' condition'])
    set(gca, 'clim', [-1 1]);
    
    % clear all temp variables 
    clear EEG_temp; clear narrowFilt; clear  matS; clear  matR; clear  covmatS; 
    clear covmatR; clear evecs; clear evals; clear sidx;clear gedData;
    
end

% save the GED data 
% filePath = 'L:\Cloud\Calypso\GED\gedData.mat';
% save(filePath, 'gedWeights');

%% plot topoplots 
event2plot = 2;
topoplot(gedWeights(event2plot).data', EEG.chanlocs, 'electrodes', 'on', 'chaninfo', EEG.chaninfo);


%% sanity check
x = 2;
figure;
plot(gedWeights(x).f, mean(gedWeights(x).psd,2));
set(gca, 'xlim', [5 45]);

%% Create Features (X) and Labels (y)

% create feature vector (X)
% the PSD of the GED filtered data
psd_20 = gedWeights(1).psd;
psd_15 = gedWeights(2).psd;

psd_20R = gedWeights(3).psd;
psd_15R = gedWeights(4).psd;

% concatenating both the metrics
psd_concatenated = [psd_20, psd_15];
% psd_concatenated_all = [psd_20, psd_20R, psd_15, psd_15R];
% create transpose (n_samples x n_features)
features_X = psd_concatenated';
% features_X = psd_concatenated_all';

% save feature vector
filePath = 'L:\Cloud\NeuroCFN\RESEARCH PROJECT\Research Project 02\Classification\Data\gedPSD_Features_X.mat';
save(filePath, 'features_X')

% create labels (y)
labels_y = [20*ones(1,size(psd_20,2)), 15*ones(1,size(psd_15,2))];
% labels_y = [20*ones(1,size(psd_20,2)), 20*ones(1,size(psd_20R,2)), 15*ones(1,size(psd_15,2)), 15*ones(1,size(psd_15R,2))];
% save labels
filePath = 'L:\Cloud\NeuroCFN\RESEARCH PROJECT\Research Project 02\Classification\Data\gedPSD_Labels_y.mat';
save(filePath, 'labels_y')





















