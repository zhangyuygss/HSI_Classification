%% plot_err_mat: plot error matrix
function [outputs] = plot_err_mat(errMatrix)
	imagesc(errMatrix); 
	colormap(flipud(gray));
	textStrings = num2str(errMatrix(:),'%0.02f');
	textStrings = strtrim(cellstr(textStrings)); 
		 
	[x,y] = meshgrid(1:size(errMatrix,1));
	hStrings = text(x(:),y(:),textStrings(:),...   
	                'HorizontalAlignment','center');
	midValue = mean(get(gca,'CLim'));  
	textColors = repmat(errMatrix(:) > midValue,1,3); 
	set(hStrings,{'Color'},num2cell(textColors,2)); 
end
