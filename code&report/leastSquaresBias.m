
function [model] = leastSquaresBias(X, y)
% compute w and bias ¦Â(assume that X_mat is invertible)
num = size(X, 1);
X_bias = [ones(num, 1), X];
para = (X_bias' * X_bias) \ X_bias' * y;
model.beta = para(1);
model.w = para(2);
model.predict = @predict;
end

function [yhat] = predict(model, Xhat)
yhat = Xhat * model.w + model.beta;
end
