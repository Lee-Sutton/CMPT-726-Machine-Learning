% polynomial_regression_1d.m
%
% Solution to Question 4.2.2

%clear variables and the screen
clear
clc
close all

% Load the data
[Countries, Features, Data] = loadUnicefData();

% Split into training and testing data
t = Data(:,2);
X = Data(:,:);

%split data x into training data and testing data
%First try 
X_train = X(1:100,:);
X_test = X(101:195,:);


%get the dimension of the input X
[~,d] = size(X_train);


%define the training set and test set
training_set = t(1:100);
test_set = t(101:195);

%Pre define my training error and testing error vectors
Training_error = zeros(1,8);
Testing_error = zeros(1,8);

%create a for loop to examine features 8-15. Store the training error and
%test error for each feature
for i = 8:15

    %Use a polynomial of degree 3
    %define the design matrix with X, using the feature at i
    Feature = X_train(:,i);
    Phi_3 = designMatrix(Feature,'polynomial',3);

    %find coefficients that maximize likelihood
    w_3 = pinv(Phi_3)*training_set;

    %predict the output values using the calculated coefficients
    %y_3 = bias + w*x + w*x^2 + w*x^3
    y_3 = w_3(1) + w_3(2)'*Feature' + w_3(3)'...
    *(Feature.^2)' + w_3(4)'*(Feature.^3)';

    %Now test the function with the test values
    Feature_test = X_test (:,i);    

    %Test the function with the test value of X
    y_3test = w_3(1) + w_3(2)*Feature_test' + w_3(3)...
    *(Feature_test.^2)' + w_3(4)*(Feature_test.^3)';

    %now calculate the training error against the training set by using the
    %Root mean square error
    %divide the root mean error by the number of data points n
    %subtract 7 from i so the index starts at 1
    [n,~] = size(training_set);    
    Training_error(i-7) = sqrt( sum((y_3-training_set').^2)/n );

    %now calculate the testing error against the testing set by using the
    %Root mean square error
    Testing_error(i-7) = sqrt( sum((y_3test-test_set').^2)/n );
    
end

%Produce a bar graph to compare the training error and testing error vs the
%input features
bar(Testing_error,'r')
hold on
bar(Training_error,0.3)
legend('Testing Error','Training Error')
ylabel('Error')
xlabel('Feature')
% set the x tick labels to features 8-15
set(gca,'XTickLabel',{8, 9, 10, 11, 12, 13, 14, 15})


%produce plots of y(x) for features 11-13
for i =11:13

    %Use a polynomial of degree 3
    %define the design matrix with X, using the feature at i
    %when plotting the data we need to make sure to sort it first
    Feature= X_train(:,i);
    Feature_test = X_test(:,i);
    Phi_3 = designMatrix(Feature,'polynomial',3);

    %find coefficients that maximize likelihood
    w_3 = pinv(Phi_3)*training_set;

    %predict the output values using the calculated coefficients
    %y_3 = bias + w*x + w*x^2 + w*x^3
    x = min(Feature_test):0.1:max(Feature_test);
    y = w_3(1) + w_3(2)'*x' + w_3(3)'...
    *(x.^2)' + w_3(4)'*(x.^3)';

    %create a new figure to plot the data
    figure
    
    %plot the training data use the unsorted data so it aligns with the
    %training set
    plot(Feature, training_set,'r*')
    hold on
    
    %plot the testing data use the unsorted data so it aligns with the
    %test set
    plot(Feature_test', test_set,'o')
    hold on
    
    %plot the predicted training function using the x and y  so it get
    %plotted in the correct order and we get a cubic function
    plot(x, y,'k','Linewidth',2)
    
    %Create the labels for the graph including legend, title, y label, and
    %x label
    %legend
    legend('Training Set','Test set','Training Function')    
    
    %Title. Ignore the underscores in the title and replace them with
    %dashes so the interpretter doesn't read them as subscripts and combine
    %with the features string to create the title
    S=strrep(Features(i), '_','-');
    graph_title = strcat(Features(1),{' vs.'},S);
    title(graph_title)
    
    %x and y label
    ylabel('Under 5 mortality rates')
    xlabel(S)  
    
end





