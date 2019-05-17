
function [model] = leastSquaresBasis(x,y,deg)
% use least squares to compute the polynomial coefficient t
%(assume that X_poly'*X_poly is invertible)
x_poly = zeros(size(x, 1), deg+1);
for i = 0:deg
    x_poly(:, i+1) = x.^i;
end
t = (x_poly' * x_poly) \ x_poly' * y;
model.t = t;
model.deg = deg;
model.predict = @predict;
end

function [yhat] = predict(model, xhat)
xhat_poly = zeros(size(xhat, 1), model.deg+1);
for i = 0:model.deg
    xhat_poly(:, i+1) = xhat.^i;
end
yhat = xhat_poly * model.t;
end
