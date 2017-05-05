%% evaluation: function description
function [errMatrix] = evaluation(true_label, estimate_label)
	cls = unique(true_label);
	clsNum = length(cls);
	errNumMatrix = zeros(clsNum, clsNum);

	for ii = 1:clsNum
		for jj = 1:clsNum
			tmp = true_label==cls(ii) & estimate_label == cls(jj);
			errNumMatrix(ii, jj) = sum(tmp);
		end
	end

	% error matrix in rate
	den = sum(errNumMatrix, 2);
	den = repmat(den, 1, clsNum);
	errMatrix = errNumMatrix./den;
end
