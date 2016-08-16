% polynomial_regression_reg.m
% Name: Lee Sutton
% Student ID: 301145106
% Solution to Question 4.4.1

% Start by clearing the screen and all variables
clear
clc
close all

% Load the data
[Countries, Features, Data] = loadUnicefData();

% Split into training and testing data
t = Data(:,2);
X = Data(:,8:end);

%normalize the data
X = normalizeData(X);

% initialize the average error to store the average error vs. lambda
avg_error = zeros(1,7);

%intialize the lambda vector
lambda = [0 0.01 0.1 1 10 100 1000 10000];

% Create a for loop to test all the different values for the regulizer
% this loop will also use cross validation to find the best value for
% lambda

for i = 1:8
    
    %initialize the RMS_Error vector for storing the validation error
    RMS_Error = zeros(10,1);
  
    %create a for loop for cross validation
    for j = 1:10
        
        %store the training values for x and the training set
        X_train = X(1:100,:);
        t_train = t(1:100,:);
        
        %store the x validation values and training validation values
        X_val = X(10*j-9:10*j,:);
        t_val = t(10*j-9:10*j,:);
        
        %delete the validation values from the training set
        X_train(10*j-9:10*j,:) = [];
        t_train(10*j-9:10*j,:) = [];
        
        %Store the dimensions of the input X
        [~,d] = size(X_train);
            
        %define the design matrix with X_train, using polynomial
        %regression of degree 2
        Phi = designMatrix(X_train,'polynomial',2);

        % find coefficients that maximize likelihood using ridge regression
        w = inv(lambda(i)*eye(2*d+1)+Phi'*Phi)*Phi'*t_train;

        % Plug in the validation points to test
        y_val = w(1) + w(2:d+1)'*X_val'...
            + w(d+2:(2*d+1))'*(X_val.^2)';

        % now calculate the validation error
        % use the Root mean square error
        % divide the root mean error by the number of data points n
        [n,~] = size(t_val);
        RMS_Error(j) = sqrt( sum((y_val-t_val').^2)/n );
        
    end
    
    %store the average value of the RMS error over all of the cross
    %validation
    avg_error(i) = mean(RMS_Error);   
    
end

% plot the validation error vs. lambda on a semilog plot
semilogx(lambda,avg_error,'-*')

%add labels to the graph
title('Average Validation Error vs. Regularization Coefficients')
xlabel('lambda')
ylabel('Validation Error')

%print the validation error for lambda = 0 on the screen since it will not
%be plotted in the log plot
avg_error(1)




    