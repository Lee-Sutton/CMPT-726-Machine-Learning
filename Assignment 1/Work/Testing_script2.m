%clear all
clear

% Load the data
[Countries, Features, Data] = loadUnicefData();

% Split into training and testing data
t = Data(:,2);
X = Data(:,8:end);
%X = normalizeData(X);

% TO DO:: Fill in
%first clear the screen
clc;
% TO DO:: Fill in
%first clear the screen
clc;
%First try polynomial regression of degree 1

%define the design matrix with X, using polynomial regression of degree 1
Phi = Testing_script(X,'polynomial',1);

%define the training set
t_set = t(1:100);
%find coefficients that maximize likelihood
w_1 = pinv(Phi)*t_set;

%predict the output values using the calculated coefficients
%ignore the first column of w_1 in the multiplication as is includes the
%bias and is a constant value, it does not multiply by any of the x terms
y_1 = w_1'*X(1:100,:)'

%now calculate the training error against the training set by using the
%Root mean square error
%divide the root mean error by the number of data points n
[n,~] = size(t_set);    
error_1 = sqrt( sum((y_1-t_set').^2)/n )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Next try a polynomial degree 2
%define the design matrix with X, using polynomial regression of degree 2
Phi_2 = Testing_script(X,'polynomial',2)

%define the training set
t_set = t(1:100);
%find coefficients that maximize likelihood
w_2 = pinv(Phi_2)*t_set;

%predict the output values using the calculated coefficients
%ignore the first column of w_2 in the multiplication as is includes the
%bias and is a constant value, it does not multiply by any of the x terms
y_2 = w_2'*X(1:100,:)'

%now calculate the training error against the training set by using the
%Root mean square error
%divide the root mean error by the number of data points n
[n,~] = size(t_set);    
error_2 = sqrt( sum((y_2-t_set').^2)/n )