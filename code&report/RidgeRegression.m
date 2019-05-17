
% Clear variables, close figures and clear command windows
clear;
close all;
clc;

% Definition of constants
num_patients = 97;
num_vars = 9;
num_train = 50;
num_test = num_patients - num_train;
num_delta = 31;

% Load data
filename = '../data/prostate.data.txt';
fileID = fopen(filename);
names = textscan(fileID, '%s', num_vars);
data = textscan(fileID, '%f');
data = reshape(data{1}, num_vars, num_patients)';
X = data(1:num_patients, 1:num_vars-1);
y = data(1:num_patients, num_vars);

% Use mask index to shuffle the orders of patients
mask = randperm(num_patients)';
X = X(mask, :);
y = y(mask, :);

% Split it into training data and test data
X_train = X(1:num_train, :);
y_train = y(1:num_train, :);
X_test = X(num_train+1:num_patients, :);
y_test = y(num_train+1:num_patients, :);

% standardization
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
    ldelta2 = -2 + 0.2 * (i - 1);  % lg(¦Ä^2) range from -2 to 4
    delta2(i) = 10 ^ (ldelta2);  % ¦Ä^2 range from 10^(-2) to 10^(4)
    theta(i,:) = ridge(X_train, y_train, delta2(i));
end

% Compute the bias term ¦Â_0 and yhat on training and test set
beta0 = mean(y_train - X_train * theta'); % we can see that ¦Â_0 are close 
                                          % to zero(details in the report),
                                          % which can be omitted
yhat_train = X_train * theta';
yhat_test = X_test * theta';

% Compute MSE on training and test set
error_train = mean((yhat_train - y_train).^2);
error_test = mean((yhat_test - y_test).^2);

% Plot theta and log10(delta2)
semilogx(delta2, theta);
legend(names{1}{1:num_vars - 1});
xlabel('¦Ä^2');
ylabel('¦È');

% Plot training and test error
figure(2);
semilogx(delta2, error_train, 'b-o');
hold on
semilogx(delta2, error_test, 'g-^');
title('Training and test error');
xlabel('¦Ä^2');
ylabel('error');
legend('Train','Test');

fclose(fileID);
