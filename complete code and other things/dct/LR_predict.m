function [labels] = LR_predict (x, W, b)
%x is suppose to be 784 X n
%W is sippose to be 784 x 10
class_num = size(W, 2);
num_test = size(x, 2);

p = exp(-W' * x - repmat(b, 1, num_test)) ./ repmat(sum(exp(-W' * x - repmat(b, 1, num_test))), class_num, 1);

[~, ind] = max(p,[], 1);
labels = (ind - 1)';