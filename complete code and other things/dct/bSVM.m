function [y_test] = bSVM(X_train, y_train, x_test, l1, l2)

dim_train = size(X_train);
dim_test = size(x_test);

num_train = dim_train(2);
num_test = dim_test(2);
orig_y = y_train;
y_train(orig_y == l1) = 1;
y_train(orig_y == l2) = -1;

K = X_train' * X_train;
alpha = zeros(num_train, 1);
max_iter = num_train * 40;
rnd = randi([1, num_train], max_iter, 1);
lambda = 1 / (64 * num_train);
avg_alpha = alpha;
step_size = 0.001;
for i = 1:max_iter
    grad = -y_train(rnd(i)) * K(rnd(i), :)' * (y_train(rnd(i)) * K(rnd(i), :) * alpha < 1) + lambda * K * alpha;
    alpha = alpha - step_size / sqrt(i) * grad;
    avg_alpha = avg_alpha * (i - 1) / i + alpha;
end

y_confidence = zeros(num_test, 1);
y_test = zeros(num_test, 1);
for j = 1:num_test
    y_confidence(j) = x_test(:, j)' *  X_train * avg_alpha;
    if (y_confidence(j) > 0) 
        y_test(j) = l1;
    else
        y_test(j) = l2;
    end
end
