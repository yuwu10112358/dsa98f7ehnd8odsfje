images_train = loadMNISTImages('../data/train-images.idx3-ubyte');
labels_train = loadMNISTLabels('../data/train-labels.idx1-ubyte');
images_test = loadMNISTImages('../data/t10k-images.idx3-ubyte');
labels_test = loadMNISTLabels('../data/t10k-labels.idx1-ubyte');

X_train = images_train(:,1:1000);
y_train = labels_train(1:1000);
x_test = images_test(:, 1:1000);
y_test = labels_test(1:1000);
num_class = 10;
init_X = images_train(:, 50001:(50000 + 100));
max_iter = 4;
factor = 4;
w = 28;
h = 28;
w_factor = 2;
h_factor = 2;
% 
[W_oracle, b_oracle] = LR_Train_Oracle(X_train, y_train, num_class);
labels_LR_oracle = LR_predict(x_test, W_oracle, b_oracle);
[X_sub_LR, label_sub_LR, sizes_LR] = generate_LR_sub_Dataset_for_LR(W_oracle, b_oracle, init_X, max_iter);
succ_LR = test_LR_sub(X_sub_LR, label_sub_LR, sizes_LR, x_test, labels_LR_oracle);
[X_sub_LR_PCA, label_sub_LR_PCA, sizes_LR_PCA] = generate_LR_sub_Dataset_for_LR_with_PCA(W_oracle, b_oracle, init_X, factor, max_iter);
succ_LR_PCA = test_LR_sub_with_PCA(X_sub_LR_PCA, label_sub_LR_PCA, sizes_LR_PCA, x_test, labels_LR_oracle, factor);
[X_sub_LR_DCT, label_sub_LR_DCT, sizes_LR_DCT] = generate_LR_sub_Dataset_for_LR_with_DCT(W_oracle, b_oracle, init_X, max_iter, w, h, w_factor, h_factor);
succ_LR_DCT = test_LR_sub_with_DCT(X_sub_LR_DCT, label_sub_LR_DCT, sizes_LR_DCT, x_test, labels_LR_oracle, w, h, w_factor, h_factor);


model_arr = SVM_Train(X_train, y_train, num_class);
labels_SVM_oracle = SVM_predict(model_arr, x_test, 10);
[X_sub_SVM, label_sub_SVM, sizes_SVM] = generate_LR_sub_Dataset_for_SVM(model_arr, init_X, max_iter);
succ_SVM = test_LR_sub(X_sub_SVM, label_sub_SVM, sizes_SVM, x_test, labels_SVM_oracle);
[X_sub_SVM_PCA, label_sub_SVM_PCA, sizes_SVM_PCA] = generate_LR_sub_Dataset_for_SVM_with_PCA(model_arr, init_X, factor, max_iter);
succ_SVM_PCA = test_LR_sub_with_PCA(X_sub_SVM_PCA, label_sub_SVM_PCA, sizes_SVM_PCA, x_test, labels_SVM_oracle, factor);
[X_sub_SVM_DCT, label_sub_SVM_DCT, sizes_SVM_DCT] = generate_LR_sub_Dataset_for_SVM_with_DCT(model_arr, init_X, max_iter, w, h, w_factor, h_factor);
succ_SVM_DCT = test_LR_sub_with_DCT(X_sub_SVM_DCT, label_sub_SVM_DCT, sizes_SVM_DCT, x_test, labels_SVM_oracle, w, h, w_factor, h_factor);
% 
labels_kNN_oracle = knn_slow(X_train, y_train, x_test);
[X_sub_kNN, label_sub_kNN, sizes_kNN] = generate_LR_sub_Dataset_for_kNN(X_train, y_train, images_train(:, 50001:(50000 + 100)), max_iter);
succ_kNN = test_LR_sub(X_sub_kNN, label_sub_kNN, sizes_kNN, x_test, labels_kNN_oracle);
[X_sub_kNN_PCA, label_sub_kNN_PCA, sizes_kNN_PCA] = generate_LR_sub_Dataset_for_kNN_with_PCA(X_train, y_train, init_X, factor, max_iter);
succ_kNN_PCA = test_LR_sub_with_PCA(X_sub_kNN_PCA, label_sub_kNN_PCA, sizes_kNN_PCA, x_test, labels_kNN_oracle, factor);
[X_sub_kNN_DCT, label_sub_kNN_DCT, sizes_kNN_DCT] = generate_LR_sub_Dataset_for_kNN_with_DCT(X_train, y_train, init_X, max_iter, w, h, w_factor, h_factor);
succ_kNN_DCT = test_LR_sub_with_DCT(X_sub_kNN_DCT, label_sub_kNN_DCT, sizes_kNN_DCT, x_test, labels_kNN_oracle, w, h, w_factor, h_factor);

% plot(0:max_iter, succ_LR_PCA, 'cyan', ...
%     0:max_iter, succ_SVM_PCA, 'g', ...
%     0:max_iter, succ_kNN_PCA, 'k');
% legend('LR Oracle', 'SVM Oracle', 'kNN Oracle');
% xlabel('Iteration');
% ylabel('Percentage Matched');


