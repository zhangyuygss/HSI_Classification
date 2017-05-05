%% HSI classification main code
clear;
row = 145; col = 145;
load data.mat;
maxround = 10;
pointNum = length(label);
clsNum = 16;
bandnum = 200;
kernel = 'RBF';  Gamma = 1;
K0 = 25;			% Sparse level
method = 'KSOMP';
windowSOMP = 4;

% Nomalize 
clsAll = nomalize(clsAll);
% clsAll = mat2gray(clsAll);

for roundnum = 1:maxround % Train round
	disp(roundnum);
	tic;
	trainIdx = [];
	for ii = 1:clsNum
		indextmp = find(label==ii);
		thisNum = length(indextmp);
		% Random select 20% points per class for training
		thisTrIdx = round(rand(round(thisNum*0.1), 1)*(thisNum-1) + min(indextmp));
		trainIdx = [trainIdx; thisTrIdx];
	end 
% 	load trainIdx.mat;
	testIdx = setdiff(1:pointNum, trainIdx);
	trData = clsAll;  trData(:,testIdx) = [];
	trLabel = label;  trLabel(:,testIdx) = [];
	trPos = posAll;  trPos(:,testIdx) = [];
	teData = clsAll;  teData(:,trainIdx) = [];
	teLabel = label;  teLabel(:,trainIdx) = [];
	tePos = posAll;  tePos(:,trainIdx) = [];

	
	K_a = compute_kernel(Gamma, trData, trData);

	% Get sparse representation of test points
	teNum = length(teLabel);
	inferCls = zeros(teNum, 1);
	inferNum = teNum;
	for ii = 1:inferNum  %length(label)
		if(~mod(ii,1000))
			disp(ii);
		end
        % disp(ii)
        if(strcmp(method, 'KOMP'))
        	x = teData(:, ii);
        	k_xx = 1;
        else
        	thisPos = tePos(:, ii);
        	maxRow = min(thisPos(1)+windowSOMP, row);
        	minRow = max(thisPos(1)-windowSOMP, 1);
        	maxCol = min(thisPos(2)+windowSOMP, col);
        	minCol = max(thisPos(2)-windowSOMP, 1);
        	x = im(minRow:maxRow, minCol:maxCol, :);
        	boxSize = size(x);
        	x = reshape(x, [boxSize(1)*boxSize(2), bandnum])';
        	x = nomalize(x);
        	k_xx = compute_kernel(Gamma, x, x);
        end
		k_ax = compute_kernel(Gamma, trData, x);
		[sparseIdx] = KOMP(K_a, k_ax, K0);
		coefficients = pinv(K_a(sparseIdx,sparseIdx))*k_ax(sparseIdx,:);
		allclsIdx = find(sparseIdx==1);	 % None zero coefficient index
		% compute class residual, infer class
		clsRes = zeros(clsNum, 1);	% Residual of clsNum classes
		tmp = 1;
		for jj = 1:clsNum
			idxtmp = find(trLabel==jj);
			thisClsIdx = intersect(allclsIdx, idxtmp); % Position index
			thisClsNum = length(thisClsIdx);
			if(thisClsNum > 0)
				thisClsCoef = coefficients(tmp:(tmp+thisClsNum-1),:);
				tmp = tmp+thisClsNum;
				res = k_xx - 2*thisClsCoef'*k_ax(thisClsIdx,:)...
					+ thisClsCoef'*K_a(thisClsIdx,thisClsIdx)*thisClsCoef;
				clsRes(jj) = norm(res, 'fro');
			else
				clsRes(jj) = Inf;
			end
		end
		[~,xclass] = min(clsRes);
		inferCls(ii) = xclass;
	end
	inferCls = inferCls';
	% Evaluation
	% Show figure
	inferGis = zeros(row, col);
	for ii = 1:inferNum
		pos = tePos(:, ii);
		inferGis(pos(1),pos(2)) = inferCls(ii);
	end
	for ii = 1:length(trLabel)
		pos = trPos(:, ii);
		inferGis(pos(1),pos(2)) = trLabel(ii);
	end
	subplot(1,2,1); imshow(label2rgb(imGIS)); title('Ground Truth');
	subplot(1,2,2); imshow(label2rgb(inferGis)); title('Estimation');

	% Error matrix
	[errMatrix] = evaluation(teLabel, inferCls);
	errMatAll{roundnum} = errMatrix;

	toc;
	% save(['classificationRst_round_' num2str(roundnum) '.mat']);
end

tmpSum = zeros(size(errMatrix));
for ii = 1:maxround
	tmpSum = tmpSum + errMatAll{ii};
end
errMatAvg = tmpSum/maxround
save HSIC_rst.mat;

% plot colormap
imagesc(errMatAvg);
colormap(gray);
