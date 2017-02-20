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
% 
[W_oracle, b_oracle] = LR_Train_Oracle(X_train, y_train, num_class);
labels_LR_oracle = LR_predict(x_test, W_oracle, b_oracle);


%%
% %adversarial samples with DCT transformation
% tic;
X_dct = images_train(:, 50001:(50000 + 100));
[X_sub_LR_nothing, label_sub_LR_nothing, sizes_LR] = generate_LR_sub_Dataset_for_LR(W_oracle, b_oracle, X_dct, 10);
[W_sub_LR, b_sub_LR] = LR_Train_Oracle(X_sub_LR_nothing, label_sub_LR_nothing, num_class);
% toc;
x_test_dct = x_test;
labels_sub_nothing = LR_predict(x_test_dct, W_sub_LR, b_sub_LR);
% x_temp = dct(x_test_dct);
% for i=1:size(x_temp, 2)
%     temp_img = x_temp(:, i);
%     [~, ind] = sort(abs(temp_img), 'descend');
%     j = 1;
%     while norm(temp_img(ind(1:j))) / norm(temp_img) < 0.99
%         j = j + 1;
%     end
%     x_temp(ind((j + 1): end), i) = 0;
% end
% x_test_dct = idct(x_temp);
adx_LR_nothing = Adversarial_LR(x_test_dct, y_test, W_sub_LR);
% toc;

labels_adx_LR_nothing = LR_predict(adx_LR_nothing, W_oracle, b_oracle);
% fprintf('the matching rate with oracle is %f\n', sum(labels_sub_nothing == labels_LR_oracle)/length(y_test) * 100);
fprintf('the missclassification rate with no transformation is %f\n', sum(labels_adx_LR_nothing ~= y_test)/length(y_test) * 100);


