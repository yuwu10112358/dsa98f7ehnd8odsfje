images_train = loadMNISTImages('train-images.idx3-ubyte');
labels_train = loadMNISTLabels('train-labels.idx1-ubyte');
images_test = loadMNISTImages('t10k-images.idx3-ubyte');
labels_test = loadMNISTLabels('t10k-labels.idx1-ubyte');

X_train = images_train(:, 1:50000);
y_train = labels_train(1:50000);
x_test = images_test;
y_test = labels_test;
m = size(X_train,1);
sub_X_init = images_train(:, 50001:(50000 + 100));
num_class = 10;
% 
[W_oracle, b_oracle] = LR_Train_Oracle(X_train, y_train, num_class);
labels_LR_oracle = LR_predict(x_test, W_oracle, b_oracle);


%%
% %adversarial samples without any transformation
% tic;
[X_sub_LR_nothing, label_sub_LR_nothing, sizes_LR] = generate_LR_sub_Dataset_for_LR(W_oracle, b_oracle, images_train(:, 50001:(50000 + 100)), 10);
[W_sub_LR, b_sub_LR] = LR_Train_Oracle(X_sub_LR_nothing, label_sub_LR_nothing, num_class);
% toc;
labels_sub_nothing = LR_predict(x_test, W_sub_LR, b_sub_LR);
adx_LR_nothing = Adversarial_LR_Papernot(x_test, y_test, W_sub_LR, b_sub_LR);
% toc;
labels_adx_LR_nothing = LR_predict(adx_LR_nothing, W_oracle, b_oracle);
% fprintf('the matching rate with oracle is %f\n', sum(labels_sub_nothing == labels_LR_oracle)/length(y_test) * 100);
fprintf('the missclassification rate with no transformation is %f\n', sum(labels_adx_LR_nothing ~= y_test)/length(y_test) * 100);
%%
% % just normalization
% tic;
% [X_sub_LR_norm, label_sub_LR_norm, sizes_LR] = generate_LR_sub_Dataset_for_LR_with_normalization(W_oracle, b_oracle, sub_X_init);
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
% labels_adx_LR_norm = LR_predict(un_nomalized_adx, W_oracle, b_oracle);
% fprintf('the matching rate with oracle is %f\n', sum(labels_sub_norm == labels_LR_oracle)/length(y_test) * 100);
% fprintf('the missclassification rate with normalization is %f\n', sum(labels_adx_LR_norm ~= y_test)/length(y_test) * 100);
%%
% normalization + PCA
tic;
factor = 1;
% [X_sub_LR_pca, label_sub_LR_pca, sizes_LR] = generate_LR_sub_Dataset_for_LR_with_PCA(W_oracle, b_oracle, sub_X_init, factor);
% succ_LR_pca = test_LR_sub_with_PCA(X_sub_LR_pca, label_sub_LR_pca, sizes_LR, images_test, labels_LR_oracle, factor);
% [X_sub_LR, label_sub_LR, sizes_LR] = generate_LR_sub_Dataset_for_LR(W_oracle, b_oracle, sub_X_init);
% succ_LR = test_LR_sub(X_sub_LR, label_sub_LR, sizes_LR, images_test, labels_LR_oracle);
% plot(0:9, succ_LR_pca, 'cyan', 0:9, succ_LR, 'g');
% legend('PCA', 'original');
% xlabel('Iteration');
% ylabel('Percentage Matched');
% ylim([45 80]);
% average = 1/size(X_sub_LR_pca,2)*sum(X_sub_LR_pca,2);
% var = 1/size(X_sub_LR_pca,2)* sum((X_sub_LR_pca - repmat(average, 1, size(X_sub_LR_pca,2))).^2, 2);
% X_temp = X_sub_LR_pca(var ~= 0, :);
% var_temp = var(var ~= 0);
% average_temp = average(var ~= 0);
% norm_X = (X_temp - repmat(average_temp, 1, size(X_sub_LR_pca,2))) ./repmat(sqrt(var_temp), 1, size(X_sub_LR_pca,2));
% coefs = pca(norm_X');
% m2 = size(coefs, 1);
% T = coefs(:,1:floor(m2/factor));
% X_new = T'*norm_X;
% [W_sub_LR, b_sub_LR] = LR_Train_Oracle(X_new, label_sub_LR_pca, num_class);

% norm_X_test = (x_test(var ~= 0, :) - repmat(average_temp, 1, size(x_test,2))) ./repmat(sqrt(var_temp), 1, size(x_test,2));
% X_test_new = T' * norm_X_test;
% labels_sub_pca = LR_predict(X_test_new, W_sub_LR, b_sub_LR);
% test_LR_sub(X_sub, label_sub, sizes, x_test, labels_oracle)
% adx_LR_pca = Adversarial_LR(X_test_new, y_test, W_sub_LR);
% labels_test = LR_predict(adx_LR_pca, W_sub_LR, b_sub_LR);
% fprintf('the mis rate for sub is %f\n', sum(labels_sub_pca == y_test)/length(y_test) * 100);
%adx_LR_pca = X_test_new;
% adx_LR_norm = T * adx_LR_pca;
% un_nomalized_adx = zeros(m, size(x_test, 2));
% un_nomalized_adx(var ~= 0, :) = repmat(sqrt(var_temp), 1, size(x_test, 2)) .* adx_LR_norm + repmat(average_temp, 1, size(x_test,2));
% un_nomalized_adx(var == 0, :) = repmat(average(var == 0), 1, size(x_test, 2));
% toc;
% 
% labels_adx_LR_pca = LR_predict(un_nomalized_adx, W_oracle, b_oracle);
% fprintf('the matching rate with oracle is %f\n', sum(labels_sub_pca == labels_LR_oracle)/length(y_test) * 100);
% fprintf('the missclassification rate with normalization is %f\n', sum(labels_adx_LR_pca ~= y_test)/length(y_test) * 100);

