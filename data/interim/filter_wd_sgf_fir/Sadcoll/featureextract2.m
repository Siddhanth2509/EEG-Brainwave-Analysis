% Specify the directory where your -2.mat files are located
matFilesDir = 'E:\BCI7thsem\emotionclips\filtered_data\filtered_data_mat\filter_wd_sgf_fir\Sadcoll';

% Get a list of all -2.mat files in the specified directory
matFiles = dir(fullfile(matFilesDir, '*-2.mat'));

% Initialize features_new matrix
features_new = [];

% Iterate through each -2.mat file
for j = 1:length(matFiles)
    % Load the artifact-free EEG data from the -2.mat file
    load(fullfile(matFilesDir, matFiles(j).name), 'artifact_free_data');

    datat= artifact_free_data;

    feature = zeros(32, 31);
    
    % Time domain
    
    for i=1:32
        cz= datat(:,i);
    
        feature(i,1)= mean(cz);
        feature(i,2)= median(cz);
        feature(i,3)= var(cz);
        feature(i,4)= std(cz);
        feature(i,5)= skewness(cz);
        feature(i,6)= kurtosis(cz);
        feature(i,7)= zerocrossrate(cz);
        % feature(i,8)= (cz); %number of waves
        feature(i,9)= peak2peak(cz);
            %feature(i,10)= instfreq(cz_sec,128);
    
            %function [H, DH] = hjorth(cz_sec);
    
        dx= diff(cz);
        ddx= diff(dx);
    
        varx= var(cz);
        varDx= var(dx);
        varDDx= var(ddx);
    
        feature(i,10)= varx;
        feature(i,11)= sqrt(varDx/varx);
        feature(i,12)= sqrt(varDDx/varDx)/feature(i,11);
    end
    
    % Frequency Domain .....................................
    
%     data= Clean_data;
%     data= data';
      win_size= 128;
    
    for i=1:32
    cz= datat(:,i);
    
    hamming_window= hamming(win_size);
    windowed_signal= cz(1:win_size).*hamming_window;
    
    fft_output= fft(windowed_signal);
    
    power_spectrum=abs(fft_output).^2;
    
    fs=128;
    f=(0:win_size-1)*(fs/win_size);
    
    %figure;
    %plot(fft_output);
    %plot(f, power_spectrum);
    
    mean_power=mean(power_spectrum);
    median_power=median(power_spectrum);
    variance_power= var(power_spectrum);
    std_power=std(power_spectrum);
    skewness_power= std(power_spectrum);
    kurtosis_power=kurtosis(power_spectrum);
    
    
    
    
    delta_band=[0.5 4];
    theta_band=[4 8];
    alpha_band=[8 13];
    beta_band=[13 30];
    gamma_band= [30 100];
    
    delta_power = sum(power_spectrum(f>=delta_band(1) & f<=delta_band(2)));
    theta_power = sum(power_spectrum(f>=theta_band(1) & f<=theta_band(2)));
    alpha_power = sum(power_spectrum(f>=alpha_band(1) & f<=alpha_band(2)));
    beta_power = sum(power_spectrum(f>=beta_band(1) & f<=beta_band(2)));
    gamma_power = sum(power_spectrum(f>=gamma_band(1) & f<=gamma_band(2)));
    
    
    theta_alpha_ratio=theta_power/alpha_power;
    beta_alpha_ratio=beta_power/alpha_power;
    ratio_one=(theta_power+alpha_power)/beta_power;
    theta_beta_ratio=theta_power/beta_power;
    gamma_delta_ratio= gamma_power/delta_power;
    ratio_two=(theta_power+alpha_power)/(alpha_power+beta_power);
    ratio_three=(gamma_power+beta_power)/(delta_power+alpha_power);
    
    
    
    [max_power, max_idx]=max(power_spectrum);
    peak_frequency=f(max_idx);
    
    
%     fprintf('mean power : %2f\n', mean_power);
    feature(i,13)=mean_power;
%     fprintf('median power : %2f\n', median_power);
    feature(i,14)=median_power;
%     fprintf('variance : %2f\n', variance_power);
    feature(i,15)=variance_power;
%     fprintf('std : %2f\n', std_power);
    feature(i,16)=std_power;
%     fprintf('skewness : %2f\n', skewness_power);
    feature(i,17)=skewness_power;
%     fprintf('kurtosis : %2f\n', kurtosis_power);
    feature(i,18)=kurtosis_power;
    
    
%     fprintf('Delta power : %.2f\n', delta_power);
    feature(i,19)=delta_power;
%     fprintf('Theta_power : %.2f\n', theta_power);
    feature(i,20)=theta_power;
%     fprintf('Alpha power : %.2f\n', alpha_power);
    feature(i,21)=alpha_power;
%     fprintf('Beta power : %.2f\n', beta_power);
    feature(i,22)=beta_power;
%     fprintf('gamma power : %.2f\n', gamma_power);
    feature(i,23)=gamma_power;
%     fprintf('peak frequency : %2f\n', peak_frequency);
    feature(i,24)=peak_frequency;
    
    
%     fprintf('theta to alpha ratio : %2f\n', theta_alpha_ratio);
    feature(i,25)=theta_alpha_ratio;
%     fprintf('beta to alpha ratio : %2f\n', beta_alpha_ratio);
    feature(i,26)=beta_alpha_ratio;
%     fprintf('theta to beta ratio : %2f\n', theta_beta_ratio);
    feature(i,27)=theta_beta_ratio;
%     fprintf('gamma to delta ratio : %2f\n', gamma_delta_ratio);
    feature(i,28)=gamma_delta_ratio;
%     fprintf('theta plus alpha by beta : %2f\n', ratio_one);
    feature(i,29)=ratio_one;
%     fprintf('theta plus alpha by alpha plus beta : %2f\n', ratio_two);
    feature(i,30)=ratio_two;
%     fprintf('gamma plus beta by delta plus alpha : %2f\n', ratio_three);
    feature(i,31)=ratio_three;
    
    end
    % Append features to features_new
    features_new = [features_new; feature];
    
    % Display a message for each file processed
    fprintf('Processed: %s\n', matFiles(j).name);
end
