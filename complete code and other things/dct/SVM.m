function [y_test] = SVM(X_train, y_train, x_test, num_class, y_test)

dim_train = size(X_train);
dim_test = size(x_test);

num_train = dim_train(2);
num_test = dim_test(2);

X_train = [X_train; ones(1, num_train)];
x_test = [x_test; ones(1, num_test)];

ensemble = zeros(num_test, num_class * (num_class - 1)/2);
count = 1;
for i = 1:num_class
    for j = i+1:num_class
        fprintf('computing class %d with %d \n', i, j);
        y_temp = y_train((y_train == i - 1) + (y_train == j - 1) > 0);
        X_temp = X_train(:, (y_train == i - 1) + (y_train == j - 1) > 0);
        model = fitcsvm(X_temp', y_temp);
        t = x_test(:, (y_test == i - 1) + (y_test == j - 1) > 0);
        y_t = y_test((y_test == i - 1) + (y_test == j - 1) > 0);
        y_p = predict(model,t');
        %y_p = bSVM(X_temp, y_temp, t, i-1, j-1);
        fprintf('accuracy: %f \n', sum(y_p == y_t) / sum((y_test == i - 1) + (y_test == j - 1) > 0));
        labels = predict(model,x_test');
%         labels = bSVM(X_temp, y_temp, x_test, i-1, j-1);
%         labels = bSVM(X_temp, y_temp, x_test, i-1, j-1);
        ensemble(:,count) = labels;
        count = count + 1;
    end
end
y_test = mode(ensemble,2);