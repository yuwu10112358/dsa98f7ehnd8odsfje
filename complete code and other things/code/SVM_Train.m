function [model_arr] = SVM_Train(X_train, y_train, num_class)
% model_arr is an k X 1 array where each array contains a svm model
% for a pair of labellings obtained using fitcsvm.

dim_train = size(X_train);
num_train = dim_train(2);
X_train = [X_train; ones(1, num_train)];
model_arr = cell(num_class * (num_class - 1)/2, 1);
count = 1;
for i = 1:num_class
    for j = i+1:num_class
        fprintf('training svm for class %d and %d\n', i, j);
        label1 = i - 1;
        label2 = j - 1;
        y_temp = y_train((y_train == label1) + (y_train == label2) > 0);
        X_temp = X_train(:, (y_train == label1) + (y_train == label2) > 0);
        model = fitcsvm(X_temp', y_temp);
        model_arr{count} = model;
        count = count + 1;
    end
end