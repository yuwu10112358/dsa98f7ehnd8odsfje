images_train = loadMNISTImages('../data/train-images.idx3-ubyte');
labels_train = loadMNISTLabels('../data/train-labels.idx1-ubyte');
images_test = loadMNISTImages('../data/t10k-images.idx3-ubyte');
labels_test = loadMNISTLabels('../data/t10k-labels.idx1-ubyte');
% pctg = zeros(10, 1);
% num = zeros(10, 1);
% for i = 1:50000
%     img = reshape(images_train(:, i), 28, 28);
%     t = dct2(img);
%     pctg(labels_train(i) + 1) = pctg(labels_train(i) + 1) + sum(sum(abs(t(1:21, 1:21)))) / sum(sum(abs(t)));
%     num(labels_train(i) + 1) = num(labels_train(i) + 1) + 1;
% end
% 
% coefs = getDCTCoefs(images_train(:, 1), 28, 28, 2, 2);
% img_r = getImgFromDCTCoefs(coefs, 28, 28, 2, 2);

X_train = images_train(:,1:1000);
y_train = labels_train(1:1000);
x_test = images_test(:, 1:1000);
y_test = labels_test(1:1000);
num_class = 10;
init_X = images_train(:, 50001:(50000 + 100));
max_iter = 4;

labels_kNN_oracle = knn_slow(X_train, y_train, x_test);
[X_sub_LR_PCA, label_sub_LR_PCA, sizes_LR_PCA] = generate_LR_sub_Dataset_for_kNN_with_DCT(X_train, y_train, init_X, max_iter, 28, 28, 2, 2);
succ_LR_PCA = test_LR_sub_with_DCT(X_sub_LR_PCA, label_sub_LR_PCA, sizes_LR_PCA, x_test, labels_kNN_oracle, 28, 28, 2, 2);