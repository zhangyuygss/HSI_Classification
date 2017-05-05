%% nomalize: function description
function [nData] = nomalize(data)
	vecLen = size(data, 1);
	Max = max(data, [], 1);
	Min = min(data, [], 1);
	nData = (data - repmat(Min, vecLen, 1))./(repmat(Max-Min, vecLen, 1));
end
