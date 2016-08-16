function Phi = Testing_script(X,basis,varargin)
% Phi = designMatrix(X,basis)
% Phi = designMatrix(X,'polynomial',degree)
% Phi = designMatrix(X,'sigmoid',Mu,s)
%
% Compute the design matrix for input data X
% X is n-by-d
% Mu is k-by-1

%The phi matrix will depend on what type of function we are using
%First we will create a Phi matrix using polynomial regression of the kth
%degreee
if strcmp(basis,'polynomial')    
  %k reads the degree of the polynomial
  k = varargin{1};

  %define constants for the size of our phi matrix
  n = 100;  % 100 data points  
  [~,d] = size(X);  %get the dimension of x
  
  %check if the polynomial is of degree 1
  if k == 1;
    %create the phi matrix with no extra column for bias
    Phi = zeros(n,d);  
  
    %The remaining columns will be defined as degree one polynomials for 
    %each element of x => phi(x) = x^1
    Phi(1:n,1:d) = X(1:n,:);  
      
  end
  
  %check if the polynomial is of degree 2
  if k == 2
      %create the phi matrix with dimensions n x (2d+1) this will allow for
      %2nd degree polynomial functions and one extra column for a constant
      %term (bias)
      Phi = ones(n,2*d); 
      
      %first input the linear functions phi(x) = x^1
      Phi(1:n,1:d) = X(1:n,:)
      
      %now input the degree to functions of phi(x) = x^2
      %use the dot operator to squre each term
      Phi(1:n,d+1:2*d) = X(1:n,:).^2  
  
      
  end

    %check if the polynomial is of degree 3
  if k == 3
      %create the phi matrix with dimensions n x (3d+1) this will allow for
      %3rd degree polynomial functions and one extra column for a constant
      %term (bias)
      Phi = ones(n,3*d+1); 
      
      %first input the linear functions phi(x) = x^1
      Phi(1:n,2:d+1) = X(1:n,:)
      
      %now input the second degree functions of phi(x) = x^2
      %use the dot operator to squre each term
      Phi(1:n,d+2:2*d+1) = X(1:n,:).^2 
      
      %now input third degree functions of phi(x) = x^3
      %use the dot operator to cube each term
      Phi(1:n,2*d+2:3*d+1) = X(1:n,:).^3         
      
  end

  
  
elseif strcmp(basis,'sigmoid')
  Mu = varargin{1};
  s = varargin{2};
  % TO DO:: Fill in
  Phi = []
  
  
else
  error('Unknown basis type');
end



