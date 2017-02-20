function [ y_test ] = knn_slow( X, y, X_test )
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

   % initialize the matrix of distances
   Dnew = [];
   % update the y_test label to the nearest neighbour's label
   % calculate the euclidean distance between each test sample and
   % each training sample 
   for i=1:floor(size(X_test,2)/1000)
       D = pdist2(X',(X_test(:,((i-1)*1000+1):1000*i))');
       % find the nearest neighbour indeces accross the columns of D
       [M,I] = min(D); 
       % update the classification of the test sample
       y_test(((i-1)*1000+1):1000*i) = y(I);
       i
   end
   % flip the dimensions
   y_test = y_test';
end
