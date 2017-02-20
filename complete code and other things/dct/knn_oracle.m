function [ y_test ] = knn_oracle( X, y, X_test )
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

   % update the y_test label to the nearest neighbour's label
   % calculate the euclidean distance between each test sample and
   % each training sample 
   D = pdist2(X',X_test');
   % find the nearest neighbour indeces accross the columns of D
   [M,I] = min(D); 
   % update the classification of the test sample
   y_test = y(I);
   
end
