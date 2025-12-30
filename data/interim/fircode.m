% Load raw EEG data (replace 'raw_eeg_data.mat' with your data file)
 load('sub11t2H.mat');

% Transpose the data_interval1 matrix for EEG toolbox (32 channels x 1920 epochs)
 EEGdata = data_interval1';


% Perform ICA using runica
[weights, sphere] = runica(EEGdata, 'extended', 1);

% Multiply the EEG data by the unmixing matrix to obtain ICs
ICs = weights * EEGdata;

ICs=ICs'


% Define the sampling frequency
sampling_frequency = 128;  % Hz

% Define the frequency range for band-pass filtering
low_cutoff = 0.5;  % Hz
high_cutoff = 45;  % Hz

% Design a low-pass filter
filter_order = 200;  % You can adjust this order as needed
low_pass_cutoff_norm = high_cutoff / (sampling_frequency / 2);
b_low = fir1(filter_order, low_pass_cutoff_norm, 'low');

% Design a high-pass filter
high_pass_cutoff_norm = low_cutoff / (sampling_frequency / 2);
b_high = fir1(filter_order, high_pass_cutoff_norm, 'high');

% Apply the band-pass filter to EEG data by cascading low-pass and high-pass filters
filtered_data = filtfilt(b_low, 1, filtfilt(b_high, 1, ICs));

% At this point, 'filtered_data' contains the EEG data after band-pass filtering

% Save the filtered EEG data to a new file (replace 'filtered_eeg_data.mat' with your desired filename)
save('filtered_sub11t11Hfir.mat', 'filtered_data');







% Define the parameters for the Savitzky-Golay filter
frame_length = 127;  % Adjust this parameter as needed
order = 5;  % Adjust this parameter as needed

% Apply the Savitzky-Golay filter
smoothed_signal = sgolayfilt(filtered_data, order, frame_length);

% Remove the average trend in the EEG data
artifact_free_data = filtered_data - smoothed_signal;

% At this point, 'artifact_free_data' contains the EEG data with the average trend removed



% Save the filtered EEG data to a new file (replace 'filtered_eeg_data.mat' with your desired filename)
save('sgolaysub11t1Hfir.mat', 'artifact_free_data');

















% 
% % Load the artifact-free EEG data (if not already loaded)
% % load('artifact_free_data.mat'); % Uncomment this line if data is saved in a separate file
% 
% % Reshape or flatten the data into a 1-D vector
% artifact_free_data1 = artifact_free_data(:);
% 
% % Define wavelet decomposition parameters
% mother_wavelet = 'db2';  % Daubechies 2 wavelet
% wavelet_level = 4;  % Decompose up to 4 levels
% 
% % Perform wavelet decomposition
% [c, l] = wavedec(artifact_free_data1, wavelet_level, mother_wavelet);
% 
% % Select the detailed coefficients at the third level of decomposition
% detail_coeff_third_level = detcoef(c, l, wavelet_level);
% 
% % Calculate the threshold for wavelet coefficient removal
% threshold_multiplier = 0.15;
% threshold = threshold_multiplier * std(detail_coeff_third_level);
% 
% % Apply soft thresholding to the detailed coefficients
% c_thresh = wthresh(c, 's', threshold);
% 
% % Reconstruct the artifact-free EEG signal
% artifact_removed_data = waverec(c_thresh, l, mother_wavelet);
% 
% % Reshape the artifact_removed_data into a 1920x32 matrix
% artifact_removed_data = reshape(artifact_removed_data, 1920, 32);
% 
% % At this point, 'artifact_removed_data' contains the EEG data with artifacts removed and is a 1920x32 double matrix
% 
% % Save the corrected EEG data to a new file (replace 'corrected_eeg_data.mat' with your desired filename)
% save('wavedec_sub11t11Hfir.mat', 'artifact_removed_data');





