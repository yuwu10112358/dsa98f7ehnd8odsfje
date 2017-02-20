function [labels] = SVM_predict(model_arr, x_test, num_class)

num_test = size(x_test, 2);
ensemble = zeros(num_test, num_class * (num_class - 1) / 2);
count = 1;
x_test = [x_test; ones(1, num_test)];
for i = 1:num_class
    for j = i+1:num_class
        y_p = predict(model_arr{count},x_test');
        ensemble(:, count) = y_p;
        count = count + 1;
    end
end

labels = mode(ensemble,2);