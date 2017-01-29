% images_train = loadMNISTImages('train-images.idx3-ubyte');
% labels_train = loadMNISTLabels('train-labels.idx1-ubyte');
% images_test = loadMNISTImages('t10k-images.idx3-ubyte');
% labels_test = loadMNISTLabels('t10k-labels.idx1-ubyte');
% 
X_train = images_train(:, 1:50000);
y_train = labels_train(1:50000);

sub_X_init = images_train(:, 50001:(50000 + 100));
% copression rates
m = size(X_train,1);

% normalize the images
average = 1/size(X_train,2)*sum(X_train,2);
var = 1/size(X_train,2)* sum((X_train - repmat(average, 1, size(X_train,2))).^2, 2);
zeros_ind = (var == 0);
X_temp = X_train(var ~= 0, :);
var_temp = var(var ~= 0);
average_temp = average(var ~= 0);
norm_X = (X_temp - repmat(average_temp, 1, size(X_train,2))) ./repmat(sqrt(var_temp), 1, size(X_train,2));
norm_X_test = (images_test(var ~= 0, :) - repmat(average_temp, 1, size(images_test,2))) ./repmat(sqrt(var_temp), 1, size(images_test,2)); 

norm_sub_X_init = (sub_X_init(var ~= 0, :) - repmat(average_temp, 1, size(sub_X_init,2))) ./repmat(sqrt(var_temp), 1, size(sub_X_init,2)); 


[coefs, ~, ~, ~, explained] = pca(norm_X');
[W_oracle, b_oracle] = LR_Train_Oracle(norm_X, y_train, 10);
labels_LR_oracle = LR_predict(norm_X_test, W_oracle, b_oracle);
fprintf('orig matched: %d \n', sum(labels_LR_oracle == labels_test));
[norm_sub_X, label_sub_LR, sizes_LR] = generate_LR_sub_Dataset_for_LR(W_oracle, b_oracle, norm_sub_X_init);
[W, b] = LR_Train_Oracle(norm_sub_X, label_sub_LR, 10);
adx_x = Adversarial_LR(norm_X_test, labels_test, W);
adx_label = LR_predict(adx_x, W_oracle, b_oracle);
fprintf('adversarial matched: %d \n', sum(adx_label == labels_test));

un_normalized_img = zeros(m, 1);
un_normalized_img(var ~= 0) = adx_x(:, 17) .* sqrt(var_temp) + average_temp;
un_normalized_img(var == 0) = images_test(var == 0, 17);


% f = [2];
% m2 = size(coefs, 1);
% for i = 1:length(f)
%     T = coefs(:,1:floor(m2/f(i)));
%     x_new = T'*norm_X;
%     [W_oracle, b_oracle] = LR_Train_Oracle(x_new, y_train, 10);
%     x_test_new = T'* norm_X_test;
%     labels_oracle = LR_predict(x_test_new, W_oracle, b_oracle);
%     [new_X_sub_LR, label_sub_LR, sizes_LR] = generate_LR_sub_Dataset_for_LR(W_oracle, b_oracle, T' * norm_sub_X_init);
%     [W, b] = LR_Train_Oracle(new_X_sub_LR, label_sub_LR, 10);
%     labels_sub = LR_predict(x_test_new, W, b);
%     adx_x = Adversarial_LR(x_test_new, labels_test, W);
%     adx_label = LR_predict(adx_x, W_oracle, b_oracle);
%     fprintf('num match for sub and oracle: %d for %d\n', sum(labels_sub == labels_oracle), i);
%     fprintf('num match for oracle on adversarial samples: %d for %d\n', sum(adx_label == labels_test), i);
% end
% 
% tst_img = X_train(:, 5);
% new_tst_img = zeros(m, 1);
% new_tst_img(var == 0) = tst_img(var == 0);
% new_tst_img(var ~= 0) = T * T' * tst_img(var ~= 0);



