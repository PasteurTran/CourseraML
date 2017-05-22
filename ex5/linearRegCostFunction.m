function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h = X * theta;  %calculus H
error = h - y;  %calculus error between guessed value and y (errors vector)
error_sqr = error.^2; %square errors 
sum_of_sqr_errors = sum(error_sqr);

%compute the J value
J = 1/(2*m) * sum_of_sqr_errors;

grad = sum(X'*error) * 1/ m; %reemmber that X has to be invesred 

%Cost regularization
theta(1) = 0;

sum_of_theta_sqr = sum(theta.^2);
J = J + (sum_of_theta_sqr*lambda )/ (2 * m);

%grad regulation - it is added BEFORE times by 1/m
grad = (X'*error +lambda*theta) * 1/ m;







% =========================================================================

grad = grad(:);

end
