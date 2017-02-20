function[success_rate] = test_LR_sub(X_sub, label_sub, sizes, x_test, labels_oracle)

num_iter = length(sizes);
success_rate_lr = zeros(num_iter, 1);
for i = 1:num_iter
    x_train = X_sub(:, 1:sizes(i));
    y_train = label_sub(1:sizes(i));
    [W, b] = LR_Train_Oracle(x_train, y_train, 10);
    label_lr_sub = LR_predict(x_test,W, b);
    success_rate_lr(i) = sum(label_lr_sub == labels_oracle)/length(label_lr_sub)*100;
    fprintf('Interation %d: error_lr: %f\n', i, success_rate_lr(i));
end

success_rate = success_rate_lr;
