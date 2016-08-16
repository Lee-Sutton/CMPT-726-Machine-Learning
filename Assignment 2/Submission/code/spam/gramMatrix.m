function K = gramMatrix(U,V,kernel,kernel_params)
% K = gramMatrix(U,V,'gaussian',{sigma})
%
% Compute a Gram matrix, all kernel values between sets of datapoints.
% U is N-by-R
% V is M-by-R
% K is N-by-M, K(u,v) is kernel(U(u,:),V(v,:))
% kernel is string, kernel type
% variable parameters per kernel type.

% Initialize a matrix status that returns 1 if a gram matrix was created or
% 0 if the input function was unknown
matrix_status = 0;

if strcmp(kernel,'gaussian')
    s = kernel_params{1};
    K = exp(-dist2(U,V)/(2*s^2));
    
    % Return a 1 for the matrix status informing the user that a gram
    % matrix was made
    matrix_status = 1;
end

if strcmp(kernel,'polynomial')
    % Polynomial degree
    s = kernel_params{1};
    
    % Gram Matrix
    K = 1 + (U*V').^s;
    % Return a 1 for the matrix status informing the user that a gram
    % matrix was made
    matrix_status = 1;
end

if strcmp(kernel,'sigmoid')
    % Parameters
    b = kernel_params{1};
    a = kernel_params{2};
    
    % Gram matrix
    K = tanh(a*U*V'+ b);
    % Return a 1 for the matrix status informing the user that a gram
    % matrix was made
    matrix_status = 1;
    
end

if strcmp(kernel,'exponential')
    s = kernel_params{1};
    K = exp(-dist2(U,V).^.5/(2*s^2));
    
    % Return a 1 for the matrix status informing the user that a gram
    % matrix was made
    matrix_status = 1;

end

% if no gram matrix has been made at this point, display unknown kernel
% function on the screen
if matrix_status == 0
    display('unknown kernel function')    
end



