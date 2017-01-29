% images_train = loadMNISTImages('train-images.idx3-ubyte');
% labels_train = loadMNISTLabels('train-labels.idx1-ubyte');
% images_test = loadMNISTImages('t10k-images.idx3-ubyte');
% labels_test = loadMNISTLabels('t10k-labels.idx1-ubyte');
% 
% X_train = images_train(:, 1:50000);
% y_train = labels_train(1:50000);
% x_test = images_test;
% y_test = labels_test;
% m = size(X_train,1);
% sub_X_init = images_train(:, 50001:(50000 + 100));
% num_class = 10;
% 
% model_arr = SVM_Train(X_train, y_train, num_class);
% labels_LR_oracle = SVM_predict(model_arr, x_test, num_class);
% 
% %adversarial samples without any transformation
% tic;
% [X_sub_LR_nothing, label_sub_LR_nothing, sizes_LR] = generate_LR_sub_Dataset_for_SVM(model_arr, images_train(:, 50001:(50000 + 100)));
% [W_sub_LR, b_sub_LR] = LR_Train_Oracle(X_sub_LR_nothing, label_sub_LR_nothing, num_class);
% toc;
% labels_sub_nothing = LR_predict(x_test, W_sub_LR, b_sub_LR);
% adx_LR_nothing = Adversarial_LR(x_test, y_test, W_sub_LR);
% toc;
% labels_adx_LR_nothing = SVM_predict(model_arr, adx_LR_nothing, num_class);
% fprintf('the matching rate with oracle is %f\n', sum(labels_sub_nothing == labels_LR_oracle)/length(y_test) * 100);
% fprintf('the missclassification rate with no transformation is %f\n', sum(labels_adx_LR_nothing ~= y_test)/length(y_test) * 100);

% just normalization
% tic;
% [X_sub_LR_norm, label_sub_LR_norm, sizes_LR] = generate_LR_sub_Dataset_for_SVM_with_normalization(model_arr, sub_X_init);
% average = 1/size(X_sub_LR_norm,2)*sum(X_sub_LR_norm,2);
% var = 1/size(X_sub_LR_norm,2)* sum((X_sub_LR_norm - repmat(average, 1, size(X_sub_LR_norm,2))).^2, 2);
% X_temp = X_sub_LR_norm(var ~= 0, :);
% var_temp = var(var ~= 0);
% average_temp = average(var ~= 0);
% norm_X = (X_temp - repmat(average_temp, 1, size(X_sub_LR_norm,2))) ./repmat(sqrt(var_temp), 1, size(X_sub_LR_norm,2));
% [W_sub_LR, b_sub_LR] = LR_Train_Oracle(norm_X, label_sub_LR_norm, num_class);
% norm_X_test = (x_test(var ~= 0, :) - repmat(average_temp, 1, size(x_test,2))) ./repmat(sqrt(var_temp), 1, size(x_test,2));
% labels_sub_norm = LR_predict(norm_X_test, W_sub_LR, b_sub_LR);
% adx_LR_norm = Adversarial_LR(norm_X_test, y_test, W_sub_LR);
% un_nomalized_adx = zeros(m, size(x_test, 2));
% un_nomalized_adx(var ~= 0, :) = repmat(sqrt(var_temp), 1, size(x_test, 2)) .* adx_LR_norm + repmat(average_temp, 1, size(x_test,2));
% un_nomalized_adx(var == 0, :) = repmat(average(var == 0), 1, size(x_test, 2));
% toc;
% 
% labels_adx_LR_norm = SVM_predict(model_arr, un_nomalized_adx, num_class);
% fprintf('the matching rate with oracle is %f\n', sum(labels_sub_norm == labels_LR_oracle)/length(y_test) * 100);
% fprintf('the missclassification rate with normalization is %f\n', sum(labels_adx_LR_norm ~= y_test)/length(y_test) * 100);
% 
% % normalization + PCA
tic;
factor = 2;
[X_sub_LR_pca, label_sub_LR_pca, sizes_LR] = generate_LR_sub_Dataset_for_SVM_with_PCA(model_arr, sub_X_init, factor);
average = 1/size(X_sub_LR_pca,2)*sum(X_sub_LR_pca,2);
var = 1/size(X_sub_LR_pca,2)* sum((X_sub_LR_pca - repmat(average, 1, size(X_sub_LR_pca,2))).^2, 2);
X_temp = X_sub_LR_pca(var ~= 0, :);
var_temp = var(var ~= 0);
average_temp = average(var ~= 0);
norm_X = (X_temp - repmat(average_temp, 1, size(X_sub_LR_pca,2))) ./repmat(sqrt(var_temp), 1, size(X_sub_LR_pca,2));
coefs = pca(norm_X');
m2 = size(coefs, 1);
T = coefs(:,1:floor(m2/factor));
X_new = T'*norm_X;
[W_sub_LR, b_sub_LR] = LR_Train_Oracle(X_new, label_sub_LR_pca, num_class);
norm_X_test = (x_test(var ~= 0, :) - repmat(average_temp, 1, size(x_test,2))) ./repmat(sqrt(var_temp), 1, size(x_test,2));
X_test_new = T' * norm_X_test;
labels_sub_pca = LR_predict(X_test_new, W_sub_LR, b_sub_LR);
adx_LR_pca = Adversarial_LR(X_test_new, y_test, W_sub_LR);
%adx_LR_pca = X_test_new;
adx_LR_norm = T * adx_LR_pca;
un_nomalized_adx = zeros(m, size(x_test, 2));
un_nomalized_adx(var ~= 0, :) = repmat(sqrt(var_temp), 1, size(x_test, 2)) .* adx_LR_norm + repmat(average_temp, 1, size(x_test,2));
un_nomalized_adx(var == 0, :) = repmat(average(var == 0), 1, size(x_test, 2));
toc;

labels_adx_LR_pca = SVM_predict(model_arr, un_nomalized_adx, num_class);
fprintf('the matching rate with oracle is %f\n', sum(labels_sub_pca == labels_LR_oracle)/length(y_test) * 100);
fprintf('the missclassification rate with normalization is %f\n', sum(labels_adx_LR_pca ~= y_test)/length(y_test) * 100);

