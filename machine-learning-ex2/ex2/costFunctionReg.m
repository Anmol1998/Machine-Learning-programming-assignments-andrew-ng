function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

a=X*theta;
h=sigmoid(a);
b=(-y.*log(h))-((1-y).*log(1-h));
J=((1/m)*(sum(b)))+((lambda/(2*m))*sum(theta.^2));
c=(X'*(h-y));
grad(1)=(1/m)*c(1);
g=(1/m)*(c(2:size(c))+((lambda/m).*theta(2:size(theta))));
for i =2:size(theta)
	grad(i)=g(i-1);
end;	


[J,grad]=costFunction(theta,X,y);
t=[0;theta(2:length(theta))];
J=J+(lambda/(2*m)*sum(t.^2));
grad=grad+(lambda/m)*t;

% =============================================================

end
