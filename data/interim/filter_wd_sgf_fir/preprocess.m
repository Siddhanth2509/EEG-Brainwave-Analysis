d1=featuresHS

% Drop the first row
d1(1, :) = [];

% Drop the 8th and 10th columns
d1(:, [8, 10]) = [];

% Extract the data from the table
d2 = d1{:,:};

% Standardize the data using zscore
standardizedData = zscore(d2);

% Create a new table with the standardized data
d3 = array2table(standardizedData, 'VariableNames', d1.Properties.VariableNames);
