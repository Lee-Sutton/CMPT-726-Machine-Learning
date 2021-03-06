function Phi = designMatrix(X,basis,varargin)
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
  [n,d] = size(X);  %get the dimension of x
  
  %check if the polynomial is of degree 1
  if k == 1;
    %create the phi matrix with one extra column to allow for bias
    Phi = ones(n,d+1);  
  
    %The first column will remain ones to allow for a bias (phi_0 = X^0)
    %The remaining columns will be defined as degree one polynomials for each
    %element of x
    Phi(1:n,2:d+1) = X(1:n,:);      
  end  
  
  %check if the polynomial is of degree 2
  if k == 2
      %create the phi matrix with dimensions n x (2d+1) this will allow for
      %2nd degree polynomial functions and one extra column for a constant
      %term (bias)
      Phi = ones(n,2*d+1); 
      
      %first input the linear functions phi(x) = x^1
      Phi(1:n,2:d+1) = X(1:n,:);
      
      %now input the degree to functions of phi(x) = x^2
      %use the dot operator to squre each term
      Phi(1:n,d+2:2*d+1) = X(1:n,:).^2;       
  end

  %check if the polynomial is of degree 3
  if k == 3
      %create the phi matrix with dimensions n x (3d+1) this will allow for
      %3rd degree polynomial functions and one extra column for a constant
      %term (bias)
      Phi = ones(n,3*d+1); 
      
      %first input the linear functions phi(x) = x^1
      Phi(1:n,2:d+1) = X(1:n,:);
      
      %now input the second degree functions of phi(x) = x^2
      %use the dot operator to squre each term
      Phi(1:n,d+2:2*d+1) = X(1:n,:).^2;
      
      %now input third degree functions of phi(x) = x^3
      %use the dot operator to cube each term
      Phi(1:n,2*d+2:3*d+1) = X(1:n,:).^3;     
  end
  
  %check if the polynomial is of degree 4
  if k == 4
      %create the phi matrix with dimensions n x (4d+1) this will allow for
      %4th degree polynomial functions and one extra column for a constant
      %term (bias)
      Phi = ones(n,4*d+1); 
      
      %first input the linear functions phi(x) = x^1
      Phi(1:n,2:d+1) = X(1:n,:);
      
      %now input the second degree functions of phi(x) = x^2
      %use the dot operator to squre each term
      Phi(1:n,d+2:2*d+1) = X(1:n,:).^2;
      
      %now input third degree functions of phi(x) = x^3
      %use the dot operator to cube each term
      Phi(1:n,2*d+2:3*d+1) = X(1:n,:).^3;    
      
      %now input functions of phi(x) = x^4
      %use the dot operator apply 4th power to each term
      Phi(1:n,3*d+2:4*d+1) = X(1:n,:).^4;      
  end
  
  %check if the polynomial is of degree 5
    if k == 5
      %create the phi matrix with dimensions n x (5d+1) this will allow for
      %5th degree polynomial functions and one extra column for a constant
      %term (bias)
      Phi = ones(n,5*d+1); 
      
      %first input the linear functions phi(x) = x^1
      Phi(1:n,2:d+1) = X(1:n,:);
      
      %now input the second degree functions of phi(x) = x^2
      %use the dot operator to squre each term
      Phi(1:n,d+2:2*d+1) = X(1:n,:).^2;
      
      %now input functions of phi(x) = x^3
      %use the dot operator to cube each term
      Phi(1:n,2*d+2:3*d+1) = X(1:n,:).^3;    
      
      %now input functions of phi(x) = x^4
      %use the dot operator apply 4th power to each term
      Phi(1:n,3*d+2:4*d+1) = X(1:n,:).^4;
      
      %now input functions of phi(x) = x^5
      %use the dot operator apply 5th power to each term
      Phi(1:n,4*d+2:5*d+1) = X(1:n,:).^5; 
    end
    
    %check if the polynomial is of degree 6
    if k == 6
      %create the phi matrix with dimensions n x (6d+1) this will allow for
      %6th degree polynomial functions and one extra column for a constant
      %term (bias)
      Phi = ones(n,6*d+1); 
      
      %first input the linear functions phi(x) = x^1
      Phi(1:n,2:d+1) = X(1:n,:);
      
      %now input the second degree functions of phi(x) = x^2
      %use the dot operator to squre each term
      Phi(1:n,d+2:2*d+1) = X(1:n,:).^2;
      
      %now input functions of phi(x) = x^3
      %use the dot operator to cube each term
      Phi(1:n,2*d+2:3*d+1) = X(1:n,:).^3;    
      
      %now input functions of phi(x) = x^4
      %use the dot operator apply 4th power to each term
      Phi(1:n,3*d+2:4*d+1) = X(1:n,:).^4;
      
      %now input functions of phi(x) = x^5
      %use the dot operator apply 5th power to each term
      Phi(1:n,4*d+2:5*d+1) = X(1:n,:).^5; 
      
      %now input functions of phi(x) = x^6
      %use the dot operator apply 5th power to each term
      Phi(1:n,5*d+2:6*d+1) = X(1:n,:).^6;     
    end
  
  
elseif strcmp(basis,'sigmoid')
    
  %print varargin for testing
  Mu1 = varargin{1};
  Mu2 = varargin{2};
  s = varargin{3};
  
  %define constants for the size of our phi matrix
  n = 100;  % 100 data points  
  [~,d] = size(X);  %get the dimension of x
  
  %using the sigmoid function we can generate a phi matrix
  %create the phi matrix with dimensions n x (2d+1) this will allow for
  %2 sigmoid functions and one extra column for a constant term (bias)
  Phi = ones(n,3); 

  %Now input the first sigmoid function
  Phi(1:n,2) = 1./(1+exp((Mu1-X)/s));
  
  %Now input the second sigmoid function
  %Use the built in matlab function for simoid
  Phi(1:n,3) = 1./(1+exp((Mu2-X)/s));

  
else
  error('Unknown basis type');
end
