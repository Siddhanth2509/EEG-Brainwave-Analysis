

% Assuming your dataset has 'features' and 'labels'
X = allfeaturesexp4(:,1:20);
Y = allfeaturesexp4(:,"label");

% Split the dataset into training (70%), validation (20%), and testing (10%)
rng(42); % For reproducibility
cv = cvpartition(size(X, 1), 'HoldOut', 0.3);
X_train = X(training(cv), :);
Y_train = Y(training(cv));

cv = cvpartition(size(X_train, 1), 'HoldOut', 0.25);
X_val = X_train(test(cv), :);
Y_val = Y_train(test(cv));
X_train = X_train(training(cv), :);
Y_train = Y_train(training(cv));

X_test = X(~(cv.test | cv.training), :);
Y_test = Y(~(cv.test | cv.training));

% Train SVM
svm_model = fitcecoc(X_train, Y_train);
Y_val_pred_svm = predict(svm_model, X_val);
evaluate_performance('SVM', Y_val, Y_val_pred_svm);

% Train SVM with modified kernel (modify as needed)
svm_modified_model = fitcecoc(X_train, Y_train, 'KernelFunction', 'polynomial', 'PolynomialOrder', 3);
Y_val_pred_svm_modified = predict(svm_modified_model, X_val);
evaluate_performance('SVM with Modified Kernel', Y_val, Y_val_pred_svm_modified);

% Train KNN
knn_model = fitcknn(X_train, Y_train, 'NumNeighbors', 5);
Y_val_pred_knn = predict(knn_model, X_val);
evaluate_performance('KNN', Y_val, Y_val_pred_knn);

% Train Tree
tree_model = fitctree(X_train, Y_train);
Y_val_pred_tree = predict(tree_model, X_val);
evaluate_performance('Tree', Y_val, Y_val_pred_tree);

% Train Ensemble with KNN
ens_knn_model = fitcensemble(X_train, Y_train, 'Method', 'Bag', 'Learners', templateKNN('NumNeighbors', 5));
Y_val_pred_ens_knn = predict(ens_knn_model, X_val);
evaluate_performance('Ensemble with KNN', Y_val, Y_val_pred_ens_knn);

% Train Ensemble with Tree
ens_tree_model = fitcensemble(X_train, Y_train, 'Method', 'Bag', 'Learners', templateTree());
Y_val_pred_ens_tree = predict(ens_tree_model, X_val);
evaluate_performance('Ensemble with Tree', Y_val, Y_val_pred_ens_tree);

function evaluate_performance(model_name, Y_true, Y_pred)
    % Evaluate performance metrics and plot confusion matrix
    accuracy = sum(Y_true == Y_pred) / numel(Y_true);
    conf_matrix = confusionmat(Y_true, Y_pred);

    recall = diag(conf_matrix) ./ sum(conf_matrix, 2);
    precision = diag(conf_matrix) ./ sum(conf_matrix, 1)';
    f1_score = 2 * (precision .* recall) ./ (precision + recall);

    fprintf('\nPerformance Metrics for %s:\n', model_name);
    fprintf('Accuracy: %.4f\n', accuracy);
    fprintf('Recall: %.4f, %.4f, %.4f\n', recall(1), recall(2), recall(3));
    fprintf('Precision: %.4f, %.4f, %.4f\n', precision(1), precision(2), precision(3));
    fprintf('F1 Score: %.4f, %.4f, %.4f\n\n', f1_score(1), f1_score(2), f1_score(3));

    figure;
    confusionchart(Y_true, Y_pred, 'Normalization', 'row-normalized', 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
    title(['Confusion Matrix for ' model_name]);
end
