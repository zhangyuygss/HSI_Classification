%% KSOMP: KSOMP algorithm for hyperspectral image classification
function [index] = KOMP(K_a, k_ax, K0)
	% K_a: Kernel matrix of training set  
	% k_ax:kernel matrix between train data and test point 
	% 		Nx1 for OMP NxT for SOMP
	% K0: sparse level, used for terminate loop
	lambda = 1e-5;
	maxiter = 200;
	N = size(K_a,1);
	index = false(N,1);

	[~, id] = max(abs(k_ax));
	index(id) = 1;
    [tt,po] = max(k_ax);
	if(tt==1)
		index(po)=1;
        return;
	end
	baseDim = sum(index);
	iter = 1;
	while(baseDim<=K0)	% Only use K0 for terminating, maybe consider residual later
		if(iter == maxiter)
			disp('Something wrong, cannot update support vector!')
		end
		C = k_ax - K_a(:,index)*pinv(K_a(index,index)+eye(baseDim)*lambda)*k_ax(index,:);
		normVec = C.^2;
		normVec = sum(normVec, 2);
		[~, id] = max(normVec);
		index(id) = 1;
		baseDim = sum(index);
		iter = iter + 1;
	end
end
