images_train = loadMNISTImages('../data/train-images.idx3-ubyte');
labels_train = loadMNISTLabels('../data/train-labels.idx1-ubyte');
images_test = loadMNISTImages('../data/t10k-images.idx3-ubyte');
labels_test = loadMNISTLabels('../data/t10k-labels.idx1-ubyte');

X_train = images_train(:, 1:50000);
y_train = labels_train(1:50000);
x_test = images_test;
y_test = labels_test;
m = size(X_train,1);
sub_X_init = images_train(:, 50001:(50000 + 100));
num_class = 10;

[W_oracle, b_oracle] = LR_Train_Oracle(X_train, y_train, num_class);
model_arr = SVM_Train(X_train, y_train, num_class);


factor = 4;
max_iter = 9;
w = 28;
h = 28;
w_factor = 2;
h_factor = 2;

misclass_rate_LR = zeros(1, max_iter);
misclass_rate_SVM = zeros(1, max_iter);
misclass_rate_kNN = zeros(1, max_iter);
misclass_rate_LR_PCA = zeros(1, max_iter);
misclass_rate_SVM_PCA = zeros(1, max_iter);
misclass_rate_kNN_PCA = zeros(1, max_iter);
misclass_rate_LR_DCT = zeros(1, max_iter);
misclass_rate_SVM_DCT = zeros(1, max_iter);
misclass_rate_kNN_DCT = zeros(1, max_iter);

[X_sub_LR, label_sub_LR, sizes_LR] = generate_LR_sub_Dataset_for_LR(W_oracle, b_oracle, sub_X_init, max_iter);
[X_sub_LR_PCA, label_sub_LR_PCA, sizes_LR_PCA] = generate_LR_sub_Dataset_for_LR_with_PCA(W_oracle, b_oracle, sub_X_init, factor, max_iter);
[X_sub_LR_DCT, label_sub_LR_DCT, sizes_LR_DCT] = generate_LR_sub_Dataset_for_LR_with_DCT(W_oracle, b_oracle, sub_X_init, max_iter, w, h, w_factor, h_factor);

[X_sub_SVM, label_sub_SVM, sizes_SVM] = generate_LR_sub_Dataset_for_SVM(model_arr, sub_X_init, max_iter);
[X_sub_SVM_PCA, label_sub_SVM_PCA, sizes_SVM_PCA] = generate_LR_sub_Dataset_for_SVM_with_PCA(model_arr, sub_X_init, factor, max_iter);
[X_sub_SVM_DCT, label_sub_SVM_DCT, sizes_SVM_DCT] = generate_LR_sub_Dataset_for_SVM_with_DCT(model_arr, sub_X_init, max_iter, w, h, w_factor, h_factor);

[X_sub_kNN, label_sub_kNN, sizes_kNN] = generate_LR_sub_Dataset_for_kNN(X_train, y_train, sub_X_init, max_iter);
[X_sub_kNN_PCA, label_sub_kNN_PCA, sizes_kNN_PCA] = generate_LR_sub_Dataset_for_kNN_with_PCA(X_train, y_train, sub_X_init, factor, max_iter);
[X_sub_kNN_DCT, label_sub_kNN_DCT, sizes_kNN_DCT] = generate_LR_sub_Dataset_for_kNN_with_DCT(X_train, y_train, sub_X_init, max_iter, w, h, w_factor, h_factor);

time = zeros(9, max_iter);
good_fellow = @(X, y, W) Adversarial_LR(X, y, W);
papernot = @(X, y, W) Adversarial_LR_Papernot(X, y, W);
ad_func = papernot;
for i = 1:max_iter
    
    X_sub = X_sub_LR(:,1:sizes_LR(i));
    lbl_sub = label_sub_LR(1:sizes_LR(i));
    [W_LR, b_LR] = LR_Train_Oracle(X_sub, lbl_sub, num_class);
    
    tic;
    adx_LR = zeros(size(x_test, 1), size(x_test, 2));
    for j = 1:size(x_test, 2)
        adx_LR(:, j) = ad_func(x_test(:, j), y_test(j), W_LR);
    end
    time(1, i) = toc;
    labels_adx_LR = LR_predict(adx_LR, W_oracle, b_oracle);
    misclass_rate_LR(i) = sum(labels_adx_LR ~= y_test)/length(y_test) * 100;
 
    X_sub = X_sub_SVM(:,1:sizes_SVM(i));
    lbl_sub = label_sub_SVM(1:sizes_SVM(i));
    [W_SVM, b_SVM] = LR_Train_Oracle(X_sub, lbl_sub, num_class);
    tic;
    adx_SVM = zeros(size(x_test, 1), size(x_test, 2));
    for j = 1:size(x_test, 2)
        adx_SVM(:, j) = ad_func(x_test(:, j), y_test(j), W_SVM);
    end
    adx_SVM = ad_func(x_test, y_test, W_SVM);
    time(2, i) = toc;
    labels_adx_SVM = SVM_predict(model_arr, adx_SVM, num_class);
    misclass_rate_SVM(i) = sum(labels_adx_SVM ~= y_test)/length(y_test) * 100;

    
    X_sub = X_sub_kNN(:,1:sizes_kNN(i));
    lbl_sub = label_sub_kNN(1:sizes_kNN(i));
    [W_kNN, b_kNN] = LR_Train_Oracle(X_sub, lbl_sub, num_class);
    tic;
    adx_kNN = zeros(size(x_test, 1), size(x_test, 2));
    for j = 1:size(x_test, 2)
        adx_kNN(:, j) = ad_func(x_test(:, j), y_test(j), W_kNN);
    end
    time(3, i) = toc;
    labels_adx_kNN = knn_slow(X_train, y_train, adx_kNN);
    misclass_rate_kNN(i) = sum(labels_adx_kNN ~= y_test)/length(y_test) * 100;
    
%     
end

for i = 1:max_iter
    norm_X = X_sub_LR_PCA(:,1:sizes_LR_PCA(i));
    lbl_sub = label_sub_LR_PCA(1:sizes_LR_PCA(i));
    m1 = size(norm_X, 1);
    m2 = size(norm_X, 2);
    if (m1 > m2)
        T = eye(size(norm_X, 1));
    else
        [coefs, ~, ~, ~, explained] = pca(norm_X');
        T = coefs(:,1:floor(m1/factor));
    end
    X_new = T'* norm_X;
    [W_LR, b_LR] = LR_Train_Oracle(X_new, lbl_sub, num_class);
    norm_X_test = x_test;
    tic;
    adx_LR_PCA = zeros(size(x_test, 1), size(x_test, 2));
    trans_x_test = T' * norm_X_test;
    for j = 1:size(x_test, 2)
        adx_LR_PCA(:, j) = T * ad_func(trans_x_test(:, j), y_test(j), W_LR);
    end
    time(4, i) = toc;
    labels_adx_LR_PCA = LR_predict(adx_LR_PCA, W_oracle, b_oracle);
    misclass_rate_LR_PCA(i) = sum(labels_adx_LR_PCA ~= y_test)/length(y_test) * 100;
    
    norm_X = X_sub_SVM_PCA(:,1:sizes_SVM_PCA(i));
    lbl_sub = label_sub_SVM_PCA(1:sizes_SVM_PCA(i));
    m1 = size(norm_X, 1);
    m2 = size(norm_X, 2);
    if (m1 > m2)
        T = eye(size(norm_X, 1));
    else
        [coefs, ~, ~, ~, explained] = pca(norm_X');
        T = coefs(:,1:floor(m1/factor));
    end
    X_new = T'* norm_X;
    [W_SVM, b_SVM] = LR_Train_Oracle(X_new, lbl_sub, num_class);
    norm_X_test = x_test;
    tic;
    adx_SVM_PCA = zeros(size(x_test, 1), size(x_test, 2));
    trans_x_test = T' * norm_X_test;
    for j = 1:size(x_test, 2)
        adx_SVM_PCA(:, j) = T * ad_func(trans_x_test(:, j), y_test(j), W_SVM);
    end
    adx_SVM_PCA = T * ad_func(trans_x_test, y_test, W_SVM);
    time(5, i) = toc;
    labels_adx_SVM_PCA = SVM_predict(model_arr, adx_SVM_PCA, num_class);
    misclass_rate_SVM_PCA(i) = sum(labels_adx_SVM_PCA ~= y_test)/length(y_test) * 100;
    
    norm_X = X_sub_kNN_PCA(:,1:sizes_SVM_PCA(i));
    lbl_sub = label_sub_kNN_PCA(1:sizes_SVM_PCA(i));
    m1 = size(norm_X, 1);
    m2 = size(norm_X, 2);
    if (m1 > m2)
        T = eye(size(norm_X, 1));
    else
        [coefs, ~, ~, ~, explained] = pca(norm_X');
        T = coefs(:,1:floor(m1/factor));
    end
    X_new = T'* norm_X;
    [W_kNN, b_kNN] = LR_Train_Oracle(X_new, lbl_sub, num_class);
    norm_X_test = x_test;
    tic;
    adx_kNN_PCA = zeros(size(x_test, 1), size(x_test, 2));
    trans_x_test = T' * norm_X_test;
    for j = 1:size(x_test, 2)
        adx_kNN_PCA(:, j) = T * ad_func(trans_x_test(:, j), y_test(j), W_kNN);
    end
    time(6, i) = toc;
    labels_adx_kNN_PCA = knn_slow(X_train, y_train, adx_kNN_PCA);
    misclass_rate_kNN_PCA(i) = sum(labels_adx_kNN_PCA ~= y_test)/length(y_test) * 100;
end

for i = 1:max_iter
    norm_X = X_sub_LR_DCT(:,1:sizes_LR_DCT(i));
    lbl_sub = label_sub_LR_DCT(1:sizes_LR_DCT(i));
    X_new = zeros(size(norm_X, 1) / 4, size(norm_X, 2));
    for j = 1:size(norm_X, 2)
        X_new(:, j) = getDCTCoefs(norm_X(:, j), w, h, w_factor, h_factor);
    end
    [W_LR, b_LR] = LR_Train_Oracle(X_new, lbl_sub, num_class);
    tic;
    adx_LR_DCT = zeros(size(x_test, 1), size(x_test, 2));
    for j = 1:size(x_test, 2)
        adx_LR_DCT(:, j) = getImgFromDCTCoefs(ad_func(getDCTCoefs(x_test(:, j), w, h, w_factor, h_factor), y_test(j), W_LR), w, h, w_factor, h_factor);
    end
    time(7, i) = toc;
    labels_adx_LR_DCT = LR_predict(adx_LR_DCT, W_oracle, b_oracle);
    misclass_rate_LR_DCT(i) = sum(labels_adx_LR_DCT ~= y_test)/length(y_test) * 100;
    
    norm_X = X_sub_SVM_DCT(:,1:sizes_SVM_DCT(i));
    lbl_sub = label_sub_SVM_DCT(1:sizes_SVM_DCT(i));
    X_new = zeros(size(norm_X, 1) / 4, size(norm_X, 2));
    for j = 1:size(norm_X, 2)
        X_new(:, j) = getDCTCoefs(norm_X(:, j), w, h, w_factor, h_factor);
    end
    [W_SVM, b_SVM] = LR_Train_Oracle(X_new, lbl_sub, num_class);
    tic;
    adx_SVM_DCT = zeros(size(x_test, 1), size(x_test, 2));
    for j = 1:size(x_test, 2)
        adx_SVM_DCT(:, j) = getImgFromDCTCoefs(ad_func(getDCTCoefs(x_test(:, j), w, h, w_factor, h_factor), y_test(j), W_SVM), w, h, w_factor, h_factor);
    end
    time(8, i) = toc;
    labels_adx_SVM_DCT = SVM_predict(model_arr, adx_SVM_DCT, num_class);
    misclass_rate_SVM_DCT(i) = sum(labels_adx_SVM_DCT ~= y_test)/length(y_test) * 100;
    
    norm_X = X_sub_kNN_DCT(:,1:sizes_SVM_DCT(i));
    lbl_sub = label_sub_kNN_DCT(1:sizes_SVM_DCT(i));
    X_new = zeros(size(norm_X, 1) / 4, size(norm_X, 2));
    for j = 1:size(norm_X, 2)
        X_new(:, j) = getDCTCoefs(norm_X(:, j), w, h, w_factor, h_factor);
    end
    [W_kNN, b_kNN] = LR_Train_Oracle(X_new, lbl_sub, num_class);
    tic;
    adx_kNN_DCT = zeros(size(x_test, 1), size(x_test, 2));
    for j = 1:size(x_test, 2)
        adx_kNN_DCT(:, j) = getImgFromDCTCoefs(ad_func(getDCTCoefs(x_test(:, j), w, h, w_factor, h_factor), y_test(j), W_kNN), w, h, w_factor, h_factor);
    end
    time(9, i) = toc;
    labels_adx_kNN_DCT = knn_slow(X_train, y_train, adx_kNN_DCT);
    misclass_rate_kNN_DCT(i) = sum(labels_adx_kNN_DCT ~= y_test)/length(y_test) * 100;
end
figure(1);
plot(1:max_iter, misclass_rate_LR, 'r', ...
    1:max_iter, misclass_rate_SVM, 'g', ...
    1:max_iter, misclass_rate_kNN, 'k', ...
    1:max_iter, misclass_rate_LR_PCA, '--r', ...
    1:max_iter, misclass_rate_SVM_PCA, '--g', ...
    1:max_iter, misclass_rate_kNN_PCA, '--k', ...
    1:max_iter, misclass_rate_LR_DCT, ':+r', ...
    1:max_iter, misclass_rate_SVM_DCT, ':+g', ...
    1:max_iter, misclass_rate_kNN_DCT, ':+k');
legend('LR', 'SVM', 'kNN', 'LR with PCA', 'SVM with PCA', 'kNN with PCA', 'LR with DCT', 'SVM with DCT', 'kNN with DCT')
xlabel('number of iterations');
ylabel('missclassification rate');
ylim([0, 100]);
%     
% figure(2);
% plot(1:max_iter, misclass_rate_LR_PCA, 'cyan', ...
%     1:max_iter, misclass_rate_SVM_PCA, 'g', ...
%     1:max_iter, misclass_rate_kNN_PCA, 'k');
% legend('LR', 'SVM', 'kNN')
% xlabel('number of iterations');
% ylabel('missclassification rate');
% ylim([0, 100]);
