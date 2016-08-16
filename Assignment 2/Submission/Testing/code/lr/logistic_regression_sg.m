% logistic_regression_sg.m
% Author: Lee Sutton
% Student no. 301145106
% This script will modify the logistic_regression.m to use stochastic
% gradient descent

% Start by clearing the variables and the screen
clear
clc
close all


% Maximum number of iterations.  Continue until this limit, or when error change is below tol.
max_iter=500;
tol = 0.0001;

% Wait for user when drawing plots.
wait_user = false;


% Get X1, X2
load('data.mat');

% Data matrix, with column of ones at end.
X = [X1; X2];
X = [X ones(size(X,1),1)];
% Target values, 0 for class 1 (datapoints X1), 1 for class 2 (datapoints X2).
t = [zeros(size(X1,1),1); ones(size(X2,1),1)];

% Initialize w.
w = [0.1 0 0]';

% Error values over all iterations.
e_all = [];

% intialize a random vector for stochastic gradient descent
n = randperm(200);

% Create a for loop to test the different step sizes
for eta = [0.1, 0.05, 0.03, 0.02, 0.01, 0.001, 0.0001]
    
    % Create a for loop for a maximum number of iterations
    for iter=1:max_iter
        
        % Compute output using current w on all data X.
        y = sigmoid(w'*X')';

        % e is the error, negative log-likelihood (Eqn 4.90)
        e = -sum(t.*log(y) + (1-t).*log(1-y));

        % Add this error to the end of error vector.
        e_all(end+1) = e;

        % Gradient of the error, using stochastic descent we run through
        % the training examples in random order looking at a single 
        % training example Xn
        for i = 1:200
            grad_e = (y(n(i))-t(n(i)))*X(n(i),:); 

            % Update w, *subtracting* a step in the error derivative since we're minimizing
            w_old = w;
            w = w - eta*grad_e';
            
            % Compute output using current w on all data X.
            y = sigmoid(w'*X')';
        end
        
        % e_all(end+1) = e;

        % Stop iterating if error doesn't change more than tol.
        if iter>1
            if abs(e-e_all(iter-1))<tol
            break;
            end
        end
    end
    
      % Plot error over iterations
      plot(e_all);
      hold on
      
      % Reset the e_all back to zero
      e_all = [];
      
      % Reset the w back to the initial guess
      w = [0.1 0 0]';    
     

end

% Label the axes and add a legend 
set(gca,'FontSize',15);
xlabel('Iteration');
ylabel('neg. log likelihood')
title('Minimization Using Stochastic Gradient Descent');
eta = [0.1, 0.05, 0.03, 0.02, 0.01, 0.001, 0.0001];
legend(cellstr(num2str(eta')))
