% Input and output directories
inputDirectory = 'D:\samannaya\emotionclips';
outputDirectory = 'D:\samannaya\emotionclipsnorm';

% Create the output directory if it doesn't exist
if ~exist(outputDirectory, 'dir')
    mkdir(outputDirectory);
end

% List all .mat files in the input directory
matFiles = dir(fullfile(inputDirectory, '*.mat'));

% Loop through each .mat file
for i = 1:length(matFiles)
    % Load the current .mat file
    currentFile = fullfile(inputDirectory, matFiles(i).name);
    loadedData = load(currentFile);
    
    % Find the variable name starting with 'data_interval'
    matchingFields = fieldnames(loadedData);
    dataIntervalField = find(contains(matchingFields, 'data_interval'));
    
    if ~isempty(dataIntervalField)
        % Extract the variable with the matching name
        dataIntervalVarName = matchingFields{dataIntervalField};
        dataIntervalMatrix = loadedData.(dataIntervalVarName);
        
        % Normalize the matrix
        normalizedMatrix = zscore(dataIntervalMatrix); % Replace 'normalize' with your normalization function
        
        % Save the normalized matrix to the output directory with the same filename
        outputFilename = fullfile(outputDirectory, matFiles(i).name);
        save(outputFilename, 'normalizedMatrix');
        
        fprintf('File %s processed and saved.\n', matFiles(i).name);
    else
        fprintf('File %s does not contain a variable starting with ''data_interval''. Skipping.\n', matFiles(i).name);
    end
end

disp('Normalization and saving completed.');
