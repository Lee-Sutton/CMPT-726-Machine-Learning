% polynomial_regression.m
% Author: Lee Sutton
% Student No. 301145106
% Solution to Question 4.2.1

% start by clearing the data and the screen
clear
clc
close all

% Load the data
[Countries, Features, Data] = loadUnicefData();

% Split into training and testing data
t = Data(:,2);
X = Data(:,8:end);
X = normalizeData(X);

%split data x into training data and testing data

X_train = X(1:100,:);
X_test = X(101:195,:);


%get the dimension of the input X
[~,d] = size(X_train);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%First try polynomial regression of degree 1
%define the design matrix with X, using polynomial regression of degree 1
Phi_1 = designMatrix(X_train,'polynomial',1);

%define the training set and training set
training_set = t(1:100);
test_set = t(101:195);

%find coefficients that maximize likelihood
w_1 = pinv(Phi_1)*training_set;

%predict the output values using the calculated coefficients
%y_1 = bias + w*x
y_1 = w_1(1) + w_1(2:d+1)'*X_train(1:100,:)';

%Test the function with the test values of X
y_1test = w_1(1) + w_1(2:d+1)'*X_test';

%now calculate the training error against the training set by using the
%Root mean square error
%divide the root mean error by the number of data points n
[n,~] = size(training_set); 
Training_error(1) = sqrt( sum((y_1-training_set').^2)/n )

%now calculate the testing error against the testing set by using the
%Root mean square error
Testing_error(1) = sqrt( sum((y_1test-test_set').^2)/n )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Next try a polynomial degree 2
%define the design matrix with X, using polynomial regression of degree 2
Phi_2 = designMatrix(X_train,'polynomial',2);

%define the training set
training_set = t(1:100);

%find coefficients that maximize likelihood
w_2 = pinv(Phi_2)*training_set;

%predict the output values using the calculated coefficients
%y_2 = bias + w*x + w*x^2
y_2 = w_2(1) + w_2(2:d+1)'*X_train(1:100,:)'...
    + w_2(d+2:(2*d+1))'*(X_train(1:100,:).^2)';

%Test the function with the test values of X
y_2test = w_2(1) + w_2(2:d+1)'*X_test'...
    + w_2(d+2:(2*d+1))'*(X_test.^2)';

%now calculate the training error against the training set by using the
%Root mean square error
%divide the root mean error by the number of data points n
[n,~] = size(training_set);    
Training_error(2) = sqrt( sum((y_2-training_set').^2)/n )

%now calculate the testing error against the testing set by using the
%Root mean square error
Testing_error(2) = sqrt( sum((y_2test-test_set').^2)/n )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Next try a polynomial degree 3
%define the design matrix with X, using polynomial regression of degree 3
Phi_3 = designMatrix(X_train,'polynomial',3);

%define the training set
training_set = t(1:100);

%find coefficients that maximize likelihood
w_3 = pinv(Phi_3)*training_set;

%predict the output values using the calculated coefficients
%y_3 = bias + w*x + w*x^2 + w*x^3
y_3 = w_3(1) + w_3(2:d+1)'*X_train(1:100,:)' + w_3(d+2:(2*d+1))'...
    *(X_train(1:100,:).^2)' + w_3((2*d+2):(3*d+1))'*(X_train(1:100,:).^3)';

%Test the function with the test value of X
y_3test = w_3(1) + w_3(2:d+1)'*X_test' + w_3(d+2:(2*d+1))'...
    *(X_test.^2)' + w_3((2*d+2):(3*d+1))'*(X_test.^3)';

%now calculate the training error against the training set by using the
%Root mean square error
%divide the root mean error by the number of data points n
[n,~] = size(training_set);    
Training_error(3) = sqrt( sum((y_3-training_set').^2)/n )

%now calculate the testing error against the testing set by using the
%Root mean square error
Testing_error(3) = sqrt( sum((y_3test-test_set').^2)/n )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Next try a polynomial degree 4
%define the design matrix with X, using polynomial regression of degree 4
Phi_4 = designMatrix(X_train,'polynomial',4);

%define the training set
training_set = t(1:100);

%find coefficients that maximize likelihood
w_4 = pinv(Phi_4)*training_set;

%predict the output values using the calculated coefficients
%y_4 = bias + w*x + w*x^2 + w*x^3 + w*^4
y_4 = w_4(1) + w_4(2:d+1)'*X_train(1:100,:)' + w_4(d+2:(2*d+1))'...
    *(X_train(1:100,:).^2)' + w_4((2*d+2):(3*d+1))'*(X_train(1:100,:).^3)'...
    + w_4((3*d+2):(4*d+1))'*(X_train(1:100,:).^4)';

%Test the function with the test value of X
y_4test = w_4(1) + w_4(2:d+1)'*X_test' + w_4(d+2:(2*d+1))'...
    *(X_test.^2)' + w_4((2*d+2):(3*d+1))'*(X_test.^3)'...
    + w_4((3*d+2):(4*d+1))'*(X_test.^4)';

%now calculate the training error against the training set by using the
%Root mean square error
%divide the root mean error by the number of data points n
[n,~] = size(training_set);    
Training_error(4) = sqrt( sum((y_4-training_set').^2)/n );

%now calculate the testing error against the testing set by using the
%Root mean square error
Testing_error(4) = sqrt( sum((y_4test-test_set').^2)/n )


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Next try a polynomial degree 5
%define the design matrix with X, using polynomial regression of degree 5
Phi_5 = designMatrix(X_train,'polynomial',5);

%define the training set
training_set = t(1:100);

%find coefficients that maximize likelihood
w_5 = pinv(Phi_5)*training_set;

%predict the output values using the calculated coefficients
%ignore the first column of w_2 in the multiplication as is includes the
%bias and is a constant value, it does not multiply by any of the x terms
%y_5 = bias + w*x + w*x^2 + w*x^3 + w*x^4 + w*x^5
y_5 = w_5(1) + w_5(2:d+1)'*X_train(1:100,:)' + w_5(d+2:(2*d+1))'...
    *(X_train(1:100,:).^2)' + w_5((2*d+2):(3*d+1))'...
    *(X_train(1:100,:).^3)' + w_5((3*d+2):(4*d+1))'...
    *(X_train(1:100,:).^4)' + w_5((4*d+2):(5*d+1))'...
    *(X_train(1:100,:).^5)';

%Test the function with the test value of X
y_5test = w_5(1) + w_5(2:d+1)'*X_test' + w_5(d+2:(2*d+1))'...
    *(X_test.^2)' + w_5((2*d+2):(3*d+1))'*(X_test.^3)'...
    + w_5((3*d+2):(4*d+1))'*(X_test.^4)'...
    + w_5((4*d+2):(5*d+1))'*(X_test.^5)';

%now calculate the training error against the training set by using the
%Root mean square error
%divide the root mean error by the number of data points n
[n,~] = size(training_set);    
Training_error(5) = sqrt( sum((y_5-training_set').^2)/n )

%now calculate the testing error against the testing set by using the
%Root mean square error
Testing_error(5) = sqrt( sum((y_5test-test_set').^2)/n )


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Next try a polynomial degree 6
%define the design matrix with X, using polynomial regression of degree 6
Phi_6 = designMatrix(X_train,'polynomial',6);

%define the training set
training_set = t(1:100);

%find coefficients that maximize likelihood
w_6 = pinv(Phi_6)*training_set;

%predict the output values using the calculated coefficients
%ignore the first column of w_2 in the multiplication as is includes the
%bias and is a constant value, it does not multiply by any of the x terms
%y_6 = bias + w*x + w*x^2 + w*x^3 + w*x^4 + w*x^5 + w*x^6
y_6 = w_6(1) + w_6(2:d+1)'*X_train' + w_6(d+2:(2*d+1))'...
    *(X_train.^2)' + w_6((2*d+2):(3*d+1))'*(X_train.^3)'...
    + w_6((3*d+2):(4*d+1))'*(X_train.^4)'...
    + w_6((4*d+2):(5*d+1))'*(X_train.^5)'...
    + w_6((5*d+2):(6*d+1))'*(X_train.^6)';

%Test the function with the test value of X
y_6test = w_6(1) + w_6(2:d+1)'*X_test' + w_6(d+2:(2*d+1))'...
    *(X_test.^2)' + w_6((2*d+2):(3*d+1))'*(X_test.^3)'...
    + w_6((3*d+2):(4*d+1))'*(X_test.^4)'...
    + w_6((4*d+2):(5*d+1))'*(X_test.^5)'...
    + w_6((5*d+2):(6*d+1))'*(X_test.^6)';

%now calculate the training error against the training set by using the
%Root mean square error
%divide the root mean error by the number of data points n
[n,~] = size(training_set);    
Training_error(6) = sqrt( sum((y_6-training_set').^2)/n )

%now calculate the testing error against the testing set by using the
%Root mean square error
Testing_error(6) = sqrt( sum((y_6test-test_set').^2)/n )

%Plot the Testing and training error vs the polynomial degree
degree = [1,2,3,4,5,6];
plot(degree, Training_error,'x-')
hold on
plot(degree,Testing_error,'o--')
legend('Training Error','Testing Error')
xlabel('degree')
ylabel('error')
title('Testing Error and Training Error vs. Polynomial Degree')
