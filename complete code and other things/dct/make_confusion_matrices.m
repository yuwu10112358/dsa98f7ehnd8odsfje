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

[W_oracle, b_oracle] = LR_Train_Oracle(X_train, y_train, num_class);
labels_oracle = LR_predict(x_test, W_oracle, b_oracle);

factor = 8;
max_iter = 9;

[X_sub_LR, label_sub_LR, sizes_LR] = generate_LR_sub_Dataset_for_LR(W_oracle, b_oracle, sub_X_init, max_iter);

%Without PCA
% [W_sub, b_sub] = LR_Train_Oracle(X_sub_LR, label_sub_LR, num_class);
% adx_FGS = Adversarial_LR(x_test, y_test, W_sub);
% labels_adx_FGS = LR_predict(adx_FGS, W_oracle, b_oracle);
% adx_papernot = Adversarial_LR_Papernot(x_test, y_test, W_sub);
% labels_adx_Papernot = LR_predict(adx_papernot, W_oracle, b_oracle);

%FGS + PCA
m1 = size(X_sub_LR, 1);
m2 = size(X_sub_LR, 2);
if (m1 > m2)
    T = eye(size(X_sub_LR, 1));
else
        [coefs, ~, ~, ~, explained] = pca(X_sub_LR');
    T = coefs(:,1:floor(m1/factor));
end
X_new = T'* X_sub_LR;
[W_sub_PCA, b_sub_PCA] = LR_Train_Oracle(X_new, label_sub_LR, num_class);
X_test_new = T' * x_test;
% adx_FGSPCA = T * Adversarial_LR(X_test_new, y_test, W_sub_PCA);
% labels_adx_FGSPCA = LR_predict(adx_FGSPCA, W_oracle, b_oracle);
adx_PapernotPCA = T * Adversarial_LR_Papernot(X_test_new, y_test, W_sub_PCA);
labels_adx_PapernotPCA = LR_predict(adx_PapernotPCA, W_oracle, b_oracle);

confusion_oracle = zeros(num_class);
confusion_papernot = zeros(num_class, length(y_test));


for target_class = 0:(num_class - 1)
    for pred_class = 0:(num_class - 1)
        confusion_oracle(target_class + 1,pred_class + 1)  = sum(labels_oracle == pred_class & y_test == target_class);
        confusion_papernot(target_class + 1,pred_class + 1)  = sum(labels_adx_PapernotPCA == pred_class & y_test == target_class);
    end
end

plotConfusionMatrix(confusion_oracle);
plotConfusionMatrix(confusion_papernot);

recall = sum(diag(confusion_papernot) ./ sum(confusion_oracle, 2)) / 10;
precision = sum(diag(confusion_papernot) ./ sum(confusion_oracle, 1)') / 10;
accuracy = sum(diag(confusion_papernot)) / length(y_test);
100 * [precision recall accuracy]