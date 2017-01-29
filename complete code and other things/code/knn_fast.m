function [ y_test ] = knn_fast( X, y, X_test )
% KNN Performs image classification using k-nearest neighbour where the
% k = 1 as per Papernot et al. (2016) on the test set
% Input: 
% X - training set of 50,000 images with 28x28 pixel dimensions (784 x 50000)
% y - training set labels (50000 x 1)
% X_test - test set of 10,000 images with 28x28 pixel dimensions (784 x
% 10000)
%
% Output:
% y_test - the classified labels associate with the test image set
    
%     % sort all the rows such that they are clustered according to the label
%     sorted_data = sortrows([y X'],1);
%     sorted_y = sorted_data(:,1);
%     size(sorted_y)
%     sorted_X = sorted_data(:,2:end);
    
    % initialize the matrix for clusters
    averages = zeros(max(y),size(X,1));
    k = 1;
    
    for i=min(y):max(y)
        % get the rows that have the correct labeling
        index = find(y == i);
        X_new = X(:,index);
        % take the average
        averages(k,:) = sum(X_new,2)/size(X_new,2);
        % update index
        k = k+1;
    end
    
   % update the y_test label to the nearest neighbour's label
   % calculate the euclidean distance between each test sample and
   % each training sample 
   D = pdist2(averages,X_test');
   % find the nearest neighbour indeces accross the columns of D
   [M,I] = min(D);
   % update the classification of the test sample
   y_test = (I-1)';     
end