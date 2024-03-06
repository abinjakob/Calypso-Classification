function erd = ERDfeature(trialData, binsize, base_start, base_end, fs)
% function erd = ERDfeature(trialData, binsize, base_start, base_end)
%
% This function is used to calculate Even Related Desynchronisation (ERD) 
% for Motor Imagery paradigm (MI) based on Pfurtscheller(1999). 
% The function loads calculates the ERD for the given trial data 
% 
% Inputs:
%   trialData (2D array) : data from the current trial (channels x samples) 
%   binsize (int)        : size of the bin
%   base_start (int)     : start time for the baseline period
%   base_end (int)       : end time for the baseline period
%   fs (int)             : sampling frequency    
%
% Ouput:
%   erd (3D array)       : Erd values for each traial (trials x channel x erd values)
%
% Example function call:
% erd = ERDfeature(trialData, binsize, base_start, base_end)


% window size
window = size(trialData,2)/binsize; 
% calculating time points 
tbins = [];
% loop over window
for t = 1:window
    start = 1+(t-1)*binsize;
    tbins{t} = (start:(start+binsize)-1);
end 

% squaring the amplitude to obtain power 
% loop over channels
for iChan = 1:size(trialData,1)
    % loop over bins 
    for iBin = 1:length(tbins)
        chanPower_vals(iChan, iBin, :) = trialData(iChan, tbins{iBin}).^2;
    end
end 

% averaging across time samples (period after event, A)
chanPower = mean(chanPower_vals,3);

% baseline duration
baseline_period = base_end - base_start;

% calculate ERD
% loop over channels 
for iChan = 1:size(chanPower,1)
    % calculating avg power in baseline period (reference period, R)
    baseline_avg = mean(chanPower(iChan, 1:(floor((baseline_period*fs)/binsize))));
    % calculating ERD% = ((A-R)/R)*100
    erd(iChan,:) = ((chanPower(iChan,:)-baseline_avg)/baseline_avg)*100;
end 





