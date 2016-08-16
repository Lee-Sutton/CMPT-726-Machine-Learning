% sigmoid_regression.m
%
%Solution to 4.3
%Start by clearing the data and clearing the screen
clear
clc
close all

% Load the data
[Countries, Features, Data] = loadUnicefData();

% Split into training and testing data
t = Data(:,2);
X = Data(:,:);
%X = normalizeData(X);


%split data x into training data and testing data
%First try 
X_train = X(1:100,:);
X_test = X(101:195,:);

%get the dimension of the input X
[~,d] = size(X_train);


%define the training set and test set
training_set = t(1:100);
test_set = t(101:195);

%define the design matrix with X, using only feature 11 (GNI Per Capita)
Feature= X_train(:,11);

%store the feature testing points for plotting
Test_Feature = X_test(:,11);

%Define my input parameters for the sigmoid function
Mu1 = 100;
Mu2 = 10000;
s = 2000;

%define the design matrix phi with the input parameter for a sigmoid
%function
Phi = designMatrix(Feature,'sigmoid', Mu1, Mu2, s);

%find coefficients that maximize likelihood
w = pinv(Phi)*training_set;

%predict the output values using the calculated coefficients
%y_3 = bias + w*sigmoid1(x) + w*sigmoid2(x)
y = w(1) + w(2)*1./(1+exp((Mu1-Feature)/s)) +...
    w(3)*1./(1+exp((Mu2-Feature)/s));

%Now test the function with the test values
Feature_test = X_test (:,11);    

%Test the function with the test value of X
y_test = w(1) + w(2)*1./(1+exp((Mu1-Feature_test)/s)) + ...
    w(3)*1./(1+exp((Mu2-Feature_test)/s));

%now calculate the training error against the training set by using the
%Root mean square error
%divide the root mean error by the number of data points n
[n,~] = size(training_set);    
Training_error = sqrt( sum((y-training_set).^2)/n )

%now calculate the testing error against the testing set by using the
%Root mean square error
Testing_error = sqrt( sum((y_test-test_set).^2)/n )

%plot the data
%Start by plotting the training function in a thick black line
%Initialize an x value starting at zero and going up to the max value of
%the feature being examined. This will produce a smoother curve of the
%predicted function
x = 10:0.1:max(Feature_test);
y_smooth = w(1) + w(2)*1./(1+exp((Mu1-x)/s)) +...
    w(3)*1./(1+exp((Mu2-x)/s));
plot(x,y_smooth,'k','Linewidth', 2)
hold on

%plot the training points with green stars
plot(Feature, training_set,'g*')
hold on

%plot the testing points with red o's
plot(Test_Feature, test_set,'ro')

%add labels
legend('Predicted Function using sigmoid regression','Training set',...
    'Testing set')
xlabel(Features{11})
ylabel('Under 5 mortality rate')
title('Under 5 Mortality Rate vs. GNI per Capita using sigmoid regression')



