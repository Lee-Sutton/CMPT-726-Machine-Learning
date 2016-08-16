% This script will plot the surfaces for problem 1
close all

x = -2:0.1:2;
y = -2:0.2:2;
[X,Y] = meshgrid(x,y);

Z1 = X + Y -1;
Z2 = zeros(size(X));
Z3 = -X - Y -1;
surf(X,Y,Z1)
hold on
surf(X,Y,Z2);
surf(X,Y,Z3)