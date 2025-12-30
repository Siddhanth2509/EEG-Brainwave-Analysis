% Assuming you have three cell arrays x, y, and z containing features for each category
% Each cell contains a matrix of dimensions n x 15, where n is the number of samples and 15 is the number of features

% Convert cell arrays to matrices
x_matrix = cell2mat(x);
y_matrix = cell2mat(y);
z_matrix = cell2mat(z);

% Perform ANOVA to compare means across different categories for each feature
p_values = zeros(1, 15); % Initialize array to store p-values
for i = 1:15
    [~, p_values(i)] = anova1([x_matrix(:,i), y_matrix(:,i), z_matrix(:,i)], [], 'off');
end

% Display p-values
disp('ANOVA p-values for each feature:');
disp(p_values);

% Plot box plots to visualize distributions of features across categories
figure;
for i = 1:15
    subplot(3, 5, i);
    boxplot([x_matrix(:,i), y_matrix(:,i), z_matrix(:,i)], 'Labels', {'Category X', 'Category Y', 'Category Z'});
    title(['Feature ', num2str(i)]);
    xlabel('Categories');
    ylabel('Feature Value');
end
