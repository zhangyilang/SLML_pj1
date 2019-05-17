
% Clear variables, close figures and clear command windows
clear;
close all;
clc;

% Load data
load ../data/basisData.mat % Loads X and y

% Fit least-squares model and its bias
model = leastSquaresBias(X, y);

% Compute training error
yhat_test = model.predict(model, X);
error_train = mean((yhat_test - y).^2);
fprintf('Updated training error = %.2f\n', error_train);

% Compute test error
yhat_test = model.predict(model, Xtest);
error_test = mean((yhat_test - ytest).^2);
fprintf('Updated test error = %.2f\n', error_test);

% Plot model
figure(1);
plot(X, y, 'b.');
title('Training Data');
hold on
Xhat = (min(X):0.1:max(X))'; % Choose points to evaluate the function
yhat = model.predict(model, Xhat);
plot(Xhat ,yhat, 'g');
