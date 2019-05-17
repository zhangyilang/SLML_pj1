
function [theta] = ridge(X, y, d2)
% compute theta (A' * A + d2 * I must be a invertible matrix,
% details in the report)
num_vars = size(X, 2);
theta = (X' * X + d2 * eye(num_vars)) \ X' * y;
end