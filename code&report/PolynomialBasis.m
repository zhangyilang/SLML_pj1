
% Clear variables, close figures and clear command windows
clear;
close all;
clc;

% Load data
load ../data/basisData.mat % Loads X and y

%Compute and store polynomial coefficients, training and test errors for deg = 0 through deg = 10
t_all = zeros(11);
error_train = zeros(11,1);
error_test = zeros(11,1);
for deg = 0:10
    
    num = deg + 1;
    fprintf('for deg = %d:\n',deg);
    % Fit least-squares model and its bias
    model = leastSquaresBasis(X, y, deg);
    t_all(num,1:num) = model.t; 
    
    % Compute training error
    yhat_test = model.predict(model, X);
    error_train(num) = mean((yhat_test - y).^2);
    fprintf('Training error = %.2f, ', error_train(num));
    
    % Compute test error
    yhat_test = model.predict(model, Xtest);
    error_test(num) = mean((yhat_test - ytest).^2);
    fprintf('test error = %.2f\n', error_test(num));

end

% pick out the model with the least test error
[min_error,i] = min(error_test);
model = leastSquaresBasis(X, y, i-1);

% Plot
figure(1);
plot(0:10, error_train, 'b-o', 'markerfacecolor', 'b');
hold on
plot(0:10, error_test, 'g-^', 'markerfacecolor', 'g');
title('Training error and test error');
legend('train','test');

figure(2);
plot(Xtest, ytest, 'b.');
title(['The best polynomial model: deg=',num2str(i-1)]);
hold on
Xhat = (min(X):0.1:max(X))'; % Choose points to evaluate the function
yhat = model.predict(model, Xhat);
plot(Xhat ,yhat, 'g');
