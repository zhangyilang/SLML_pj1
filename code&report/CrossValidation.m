
% Clear variables, close figures and clear command windows
clear;
close all;
clc;

% Definition of constants
num_patients = 97;
num_vars = 9;
data_per_fold = 19;
num_delta = 26;
num_folds = 5;

% Load data
filename = '../data/prostate.data.txt';
fileID = fopen(filename);
names = textscan(fileID, '%s', num_vars);
data = textscan(fileID, '%f');
data = reshape(data{1}, num_vars, num_patients)';
X = data(1:num_patients, 1:num_vars - 1);
y = data(1:num_patients, num_vars);

% Use mask index to shuffle the orders of patients and discard 2 lines of
% data randomly that we can divide the data into 5 folds evenly
mask = randperm(num_patients, num_patients - 2)';
X = X(mask, :);
y = y(mask, :);

% 5-fold cross validation
relative_error_train = zeros(num_folds, num_delta);
relative_error_test = zeros(num_folds, num_delta);
for n = 1:num_folds
    
    % Split it into training data and test data
    X_test = X(data_per_fold * (n - 1) + 1:data_per_fold * n, :);
    y_test = y(data_per_fold * (n - 1) + 1:data_per_fold * n, :);
    X_train = X([1:data_per_fold * (n - 1), data_per_fold * n + 1:num_patients - 2], :);
    y_train = y([1:data_per_fold * (n - 1), data_per_fold * n + 1:num_patients - 2], :);
    
    % Nomalization
    X_mean = mean(X_train);
    y_mean = mean(y_train);
    X_std = sqrt(mean((X_train - X_mean).^2));
    X_train = (X_train - X_mean) ./ X_std;
    y_train = y_train - y_mean;
    X_test = (X_test - X_mean) ./ X_std;
    y_test = y_test - y_mean;
    
    % Compute the ridge regression solutions for a range of regularizers (¦Ä^2)
    delta2 = zeros(num_delta, 1);
    theta = zeros(num_delta, num_vars - 1);
    for i = 1:num_delta
        ldelta2 = -2 + 0.2 * (i - 1);  % lg(¦Ä^2) range from -2 to 3
        delta2(i) = 10 ^ (ldelta2);  % ¦Ä^2 range from 10^(-2) to 10^(3)
        theta(i,:) = ridge(X_train, y_train, delta2(i));
    end
    
    % Compute yhat on training and test set
    yhat_train = X_train * theta';
    yhat_test = X_test * theta';
    
    % Compute relative error on training and test set
    relative_error_train(n, :) = sqrt(sum((yhat_train - y_train).^2) / sum(y_train.^2));
    relative_error_test(n, :) = sqrt(sum((yhat_test - y_test).^2) / sum(y_test.^2));
    
end

% Compute average relative errors in 5-fold cross validation
mean_error_train =  mean(relative_error_train);
mean_error_test =  mean(relative_error_test);

% Plot relative training and test error against ¦Ä^2
semilogx(delta2, mean_error_train, 'b-o', 'markerfacecolor', 'b');
hold on;
semilogx(delta2, mean_error_test, 'g-^', 'markerfacecolor', 'g');
legend('Train','Test');
xlabel('¦Ä^2');
ylabel('||y-X¦È||_2/||y||_2');

% Choose the value of ¦Ä^2 with the lowest test error
[minerror, loc] = min(mean_error_test);
fprintf('¦Ä^2 = %.2f has the lowest error (%.2f) on test set\n', delta2(loc), minerror);

fclose(fileID);
