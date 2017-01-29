function[success_rate] = test_LR_sub_with_PCA(X_sub, label_sub, sizes, x_test, labels_oracle, factor)

num_iter = length(sizes);
success_rate_lr = zeros(num_iter, 1);
for i = 1:num_iter
    x_train = X_sub(:, 1:sizes(i));
    y_train = label_sub(1:sizes(i));
    
    norm_X = x_train;
    
    m1 = size(norm_X, 1);
    m2 = size(norm_X, 2);
    
    if (m1 > m2)
        T = eye(size(norm_X, 1));
    else
        coefs = pca(norm_X');
        T = coefs(:,1:floor(m1/factor));
    end

    X_new = T'*norm_X;
    [W_sub_LR, b_sub_LR] = LR_Train_Oracle(X_new, y_train, 10);
    X_test_new = T' * x_test;
    labels_sub_pca = LR_predict(X_test_new, W_sub_LR, b_sub_LR);
    success_rate_lr(i) = sum(labels_sub_pca == labels_oracle)/length(labels_oracle)*100;
    fprintf('Iteration %d: error_lr: %f\n', i, success_rate_lr(i));
end

success_rate = success_rate_lr;