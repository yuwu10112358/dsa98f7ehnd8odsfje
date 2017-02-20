function[success_rate] = test_LR_sub_with_DCT(X_sub, label_sub, sizes, x_test, labels_oracle, w, h, w_factor, h_factor)

num_iter = length(sizes);
success_rate_lr = zeros(num_iter, 1);
for i = 1:num_iter
    x_train = X_sub(:, 1:sizes(i));
    y_train = label_sub(1:sizes(i));
    
    norm_X = x_train;
    
    X_new = zeros(size(norm_X, 1) / 4, size(norm_X, 2));
    for j = 1:size(norm_X, 2)
        X_new(:, j) = getDCTCoefs(norm_X(:, j), w, h, w_factor, h_factor);
    end
    [W_sub_LR, b_sub_LR] = LR_Train_Oracle(X_new, y_train, 10);
    X_test_new = zeros(size(norm_X, 1) / 4, size(x_test, 2));
    for j = 1:size(x_test, 2)
        X_test_new(:, j) = getDCTCoefs(x_test(:, j), w, h, w_factor, h_factor);
    end
    labels_sub_pca = LR_predict(X_test_new, W_sub_LR, b_sub_LR);
    success_rate_lr(i) = sum(labels_sub_pca == labels_oracle)/length(labels_oracle)*100;
    fprintf('Iteration %d: error_lr: %f\n', i, success_rate_lr(i));
end

success_rate = success_rate_lr;