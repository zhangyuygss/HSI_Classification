%% compute_kernal: compute RBF kernel of matrix with matrix
function [knl] = compute_kernal(Gamma, A, X)
	% Gamma: RBF parameter, exp(-Gamma||a_i - x_j||^2)
	% A:Matrix [a_1, a_2,...,a_n] X:Matrix [x_1,...,x_m]
	% knl: output kernel matrix 
	N = size(A, 2);
	M = size(X, 2);
	% Compute ||a_i - x_j||^2
	asq = sum(A.^2, 1);		% 1xN
	xsq = sum(X.^2, 1);		% 1xM
	asq = ones(M, 1)*asq; 	% MxN
	xsq = ones(N, 1)*xsq; 	% NxM
	sqMtrix = asq' + xsq - 2*A'*X; 	% NxM
	
	knl = exp(-Gamma*sqMtrix);
end
