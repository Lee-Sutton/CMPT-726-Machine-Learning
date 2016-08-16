% do_kernel_perceptron.m
% Author: Lee Sutton
% Student Number: 301145106
% Start by clearing the variables and clearing the screen
clear
close all
clc

MAX_EPOCH=100;
ETA=0.00001;
NORMALIZE=0;

% Set a step size eta
eta = 1;

% Run a for loop to test each kernel function 5 times then take the average
% validation error at the end
for n = 1:5

    % Positive class
    POS=1;

    % Load data
    load('email.mat');

    % Class POS versus the rest.
    % This sets up L as +/-1 for the two classes.
    L = (Ltrain == POS) - (Ltrain~= POS);

    if NORMALIZE
      tr_mean = mean(Ftrain,1);
      tr_std = std(Ftrain,1,1) + eps;
      Ftrain = Ftrain - repmat(tr_mean, [Ntrain 1]);
      Ftrain = Ftrain ./ repmat(tr_std, [Ntrain 1]);
    end

    % Split the data up for cross validation
    % Make sure to get samples from both the ham (not spam) and the spam email
    % The first 2500 values in Ftrain are ham the last 500 are spam
    % Randomly remove 100 values from Ftrain and L values for validatiopm
    % 60 from ham 40 from spam
    rand_ham = randperm(2500,60);
    rand_spam = randperm(500,40) + 2500;
    validation = [rand_ham rand_spam];

    % Set up a for loop to store these validation values and remove them from
    % the training set
    t = 1;  % Index for the validation vectors
    for i = validation
        % Assign the validation values to the validation sets
        Fval(t,:) = Ftrain(i,:);
        Lval(t) = L(i);

        t = t + 1;
    end

    % Remove the validation values from the training set
    Ftrain = removerows(Ftrain,validation);
    L = removerows(L,validation);
    Ntrain = size(Ftrain,1);

    % Kernel parameters
    K_TYPE = 'gaussian';
    K_PARAMS = {5};

    % Train kernel perceptron
    % Compute gram matrix
    K = gramMatrix(Ftrain,Ftrain,K_TYPE,K_PARAMS);

    % Run stochastic gradient descent
    % Initialize alpha to be all zeroes
    Alpha = zeros(Ntrain,1);

    for epoch=1:MAX_EPOCH
      rp = randperm(Ntrain);
      for n_i=rp
        % TO DO:: Stochastic gradient update.
        % Evaluate the function at n_i
        y = Alpha'*K(:,n_i);

        % Check the sign of y and correct Alpha if it is wrong
        if sign(y) ~= L(n_i)
            % Update the value of alpha
            Alpha(n_i) = Alpha(n_i) + eta*L(n_i);        
        end

      end

      % Debug: print out total error.
      Fn = sign((Alpha') * K)';
      nerr = sum(Fn ~= L);
      % The following print function was disabled to increase speed
      % fprintf('Epoch %d: error %d\n', epoch, nerr);

      if nerr <=0
        break
      end
    end

    % Now validate the function with the validation set
    % Compute gram matrix using the validation values with the training values
    K = gramMatrix(Ftrain,Fval,K_TYPE,K_PARAMS);
    Fn = sign((Alpha') * K);
    display('Gaussian Validation error')
    val_err(1) = sum(Fn ~= Lval);
    val_err(1)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Test the next kernel function
    % Kernel parameters
    K_TYPE = 'polynomial';
    K_PARAMS = {1};
    % Train kernel perceptron
    % Compute gram matrix for a polynomial
    K = gramMatrix(Ftrain,Ftrain,K_TYPE,K_PARAMS);

    % Run stochastic gradient descent
    % Initialize alpha to be all zeroes
    Alpha = zeros(Ntrain,1);

    for epoch=1:MAX_EPOCH
      rp = randperm(Ntrain);
      for n_i=rp
        % TO DO:: Stochastic gradient update.
        % Evaluate the function at n_i
        y = Alpha'*K(:,n_i);

        % Check the sign of y and correct Alpha if it is wrong
        if sign(y) ~= L(n_i)
            % Update the value of alpha
            Alpha(n_i) = Alpha(n_i) + eta*L(n_i);        
        end

      end

      % Debug: print out total error.
      Fn = sign((Alpha') * K)';
      nerr = sum(Fn ~= L);
      % fprintf('Epoch %d: error %d\n', epoch, nerr);

      if nerr <=0
        break
      end
    end

    % Now validate the function with the validation set
    % Compute gram matrix using the validation values with the training values
    K = gramMatrix(Ftrain,Fval,K_TYPE,K_PARAMS);
    Fn = sign((Alpha') * K);
    display('Polynomial degree 1 - Validation error')
    val_err(2) = sum(Fn ~= Lval);
    val_err(2)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Test the next kernel function
    % Try a polynomial degree 2
    % Kernel parameters
    K_TYPE = 'polynomial';
    disp('Polynomial Degree 2')
    K_PARAMS = {2};
    % Train kernel perceptron
    % Compute gram matrix for a polynomial
    K = gramMatrix(Ftrain,Ftrain,K_TYPE,K_PARAMS);

    % Run stochastic gradient descent
    % Initialize alpha to be all zeroes
    Alpha = zeros(Ntrain,1);

    for epoch=1:MAX_EPOCH
      rp = randperm(Ntrain);
      for n_i=rp
        % TO DO:: Stochastic gradient update.
        % Evaluate the function at n_i
        y = Alpha'*K(:,n_i);

        % Check the sign of y and correct Alpha if it is wrong
        if sign(y) ~= L(n_i)
            % Update the value of alpha
            Alpha(n_i) = Alpha(n_i) + eta*L(n_i);        
        end

      end

      % Debug: print out total error.
      Fn = sign((Alpha') * K)';
      nerr = sum(Fn ~= L);
      % The following command was commented out to increase speed
      % fprintf('Epoch %d: error %d\n', epoch, nerr);

      if nerr <=0
        break
      end
    end

    % Now validate the function with the validation set
    % Compute gram matrix using the validation values with the training values
    K = gramMatrix(Ftrain,Fval,K_TYPE,K_PARAMS);
    Fn = sign((Alpha') * K);
    display('Polynomial degree 2 error')
    val_err(3) = sum(Fn ~= Lval);
    val_err(3)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Test the next kernel function
    % Try a polynomial degree 5
    % Kernel parameters
    K_TYPE = 'polynomial';
    disp('Polynomial Degree 5')
    K_PARAMS = {5};
    % Train kernel perceptron
    % Compute gram matrix for a polynomial
    K = gramMatrix(Ftrain,Ftrain,K_TYPE,K_PARAMS);

    % Run stochastic gradient descent
    % Initialize alpha to be all zeroes
    Alpha = zeros(Ntrain,1);

    for epoch=1:MAX_EPOCH
      rp = randperm(Ntrain);
      for n_i=rp
        % TO DO:: Stochastic gradient update.
        % Evaluate the function at n_i
        y = Alpha'*K(:,n_i);

        % Check the sign of y and correct Alpha if it is wrong
        if sign(y) ~= L(n_i)
            % Update the value of alpha
            Alpha(n_i) = Alpha(n_i) + eta*L(n_i);        
        end

      end

      % Debug: print out total error.
      Fn = sign((Alpha') * K)';
      nerr = sum(Fn ~= L);
      % fprintf('Epoch %d: error %d\n', epoch, nerr);

      if nerr <=0
        break
      end
    end

    % Now validate the function with the validation set
    % Compute gram matrix using the validation values with the training values
    K = gramMatrix(Ftrain,Fval,K_TYPE,K_PARAMS);
    Fn = sign((Alpha') * K);
    display('Polynomial degree 5 error')
    val_err(4) = sum(Fn ~= Lval);
    val_err(4)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Test the next kernel function
    % Try a sigmoid function
    % Kernel parameters
    K_TYPE = 'sigmoid';
    disp('Sigmoid')
    K_PARAMS = {5,2};
    % Train kernel perceptron
    % Compute gram matrix for a polynomial
    K = gramMatrix(Ftrain,Ftrain,K_TYPE,K_PARAMS);

    % Run stochastic gradient descent
    % Initialize alpha to be all zeroes
    Alpha = zeros(Ntrain,1);

    for epoch=1:MAX_EPOCH
      rp = randperm(Ntrain);
      for n_i=rp
        % TO DO:: Stochastic gradient update.
        % Evaluate the function at n_i
        y = Alpha'*K(:,n_i);

        % Check the sign of y and correct Alpha if it is wrong
        if sign(y) ~= L(n_i)
            % Update the value of alpha
            Alpha(n_i) = Alpha(n_i) + eta*L(n_i);        
        end

      end

      % Debug: print out total error.
      Fn = sign((Alpha') * K)';
      nerr = sum(Fn ~= L);
      % fprintf('Epoch %d: error %d\n', epoch, nerr);

      if nerr <=0
        break
      end
    end

    % Now validate the function with the validation set
    % Compute gram matrix using the validation values with the training values
    K = gramMatrix(Ftrain,Fval,K_TYPE,K_PARAMS);
    Fn = sign((Alpha') * K);
    display('Sigmoid basis validation error')
    val_err(5) = sum(Fn ~= Lval);
    val_err(5)
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Test the next kernel function
    % Try an exponential kernel
    % Kernel parameters
    K_TYPE = 'exponential';
    disp('Exponential')
    K_PARAMS = {5};
    % Train kernel perceptron
    % Compute gram matrix for a polynomial
    K = gramMatrix(Ftrain,Ftrain,K_TYPE,K_PARAMS);

    % Run stochastic gradient descent
    % Initialize alpha to be all zeroes
    Alpha = zeros(Ntrain,1);

    for epoch=1:MAX_EPOCH
      rp = randperm(Ntrain);
      for n_i=rp
        % TO DO:: Stochastic gradient update.
        % Evaluate the function at n_i
        y = Alpha'*K(:,n_i);

        % Check the sign of y and correct Alpha if it is wrong
        if sign(y) ~= L(n_i)
            % Update the value of alpha
            Alpha(n_i) = Alpha(n_i) + eta*L(n_i);        
        end

      end

      % Debug: print out total error.
      Fn = sign((Alpha') * K)';
      nerr = sum(Fn ~= L);
      % fprintf('Epoch %d: error %d\n', epoch, nerr);

      if nerr <=0
        break
      end
    end

    % Now validate the function with the validation set
    % Compute gram matrix using the validation values with the training values
    K = gramMatrix(Ftrain,Fval,K_TYPE,K_PARAMS);
    Fn = sign((Alpha') * K);
    display('Exponential basis validation error')
    val_err(6) = sum(Fn ~= Lval);
    val_err(6)
   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %store the validation error for later comparison
    validation_error(n,:) = val_err;

    % Plot the resulting validation error vs. the basis function
    figure
    bar(val_err)
    set(gca,'XTickLabel',{'Gaussian', '(1 + x^Tx)^1', '(1 + x^Tx)^2',...
        '(1 + x^Tx)^5','sigmoid','exponential'})
    title('Validation Error vs Kernel Function')
    ylabel('Validation Error - Number of points misclassified out of 100')

% end of the for loop for cross validation
end

% Compute the final average validation errors
gauss_error = sum(validation_error(:,1))/n;
poly1_error = sum(validation_error(:,2))/n;
poly2_error = sum(validation_error(:,3))/n;
poly5_error = sum(validation_error(:,4))/n;
sigmoid_error = sum(validation_error(:,5))/n;

% Plot this average validation error
figure
bar(mean(validation_error))
set(gca,'XTickLabel',{'Gaussian', '(1 + x^Tx)^1', '(1 + x^Tx)^2',...
    '(1 + x^Tx)^5','sigmoid','exponential'})
title('Average Validation Error vs Kernel Function')
ylabel('Average Validation Error (# of points misclassified out of 100)')



% Test kernel perceptron
% Use the exponential function
% Kernel parameters
K_TYPE = 'exponential';
disp('Exponential')
K_PARAMS = {5};

% Reload the data and train on Ftrain without removing any cross validation
% data
load('email.mat');

% Class POS versus the rest.
% This sets up L as +/-1 for the two classes.
L = (Ltrain == POS) - (Ltrain~= POS);

if NORMALIZE
  tr_mean = mean(Ftrain,1);
  tr_std = std(Ftrain,1,1) + eps;
  Ftrain = Ftrain - repmat(tr_mean, [Ntrain 1]);
  Ftrain = Ftrain ./ repmat(tr_std, [Ntrain 1]);
end

% Train kernel perceptron
% Compute gram matrix for a polynomial
K = gramMatrix(Ftrain,Ftrain,K_TYPE,K_PARAMS);

% Run stochastic gradient descent
% Initialize alpha to be all zeroes
Ntrain = size(Ftrain,1);
Alpha = zeros(Ntrain,1);

for epoch=1:MAX_EPOCH
  rp = randperm(Ntrain);
  for n_i=rp
    % TO DO:: Stochastic gradient update.
    % Evaluate the function at n_i
    y = Alpha'*K(:,n_i);

    % Check the sign of y and correct Alpha if it is wrong
    if sign(y) ~= L(n_i)
        % Update the value of alpha
        Alpha(n_i) = Alpha(n_i) + eta*L(n_i);        
    end

  end

  % Debug: print out total error.
  Fn = sign((Alpha') * K)';
  nerr = sum(Fn ~= L);
  % fprintf('Epoch %d: error %d\n', epoch, nerr);

  if nerr <=0
    break
  end
end
    
% Load data
load('test.mat')

Ntest = size(Ftest,1);
if NORMALIZE
  % Normalize the test data
  Ftest = Ftest - repmat(tr_mean, [Ntest 1]);
  Ftest = Ftest ./ repmat(tr_std, [Ntest 1]);
end


% TO DO:: Evaluate on test data.
Fn = ones(Ntest,1);

% Input the test data into the trained function
K = gramMatrix(Ftrain,Ftest,K_TYPE,K_PARAMS);
Fn = sign((Alpha') * K);

email = 'lmsutton@sfu.ca';
save('spamtest.mat','Fn','email');



