function [y_test] = LR (X_train, y_train, x_test, num_class)

dim_train = size(X_train);
dim_test = size(x_test);

image_s = dim_train(1);
num_train = dim_train(2);
num_test = dim_test(2);
X_train = [X_train; ones(1, num_train)]';
x_test = [x_test; ones(1, num_test)]';
y_train = onehot(y_train, num_class);
w = zeros(image_s + 1, num_class);

max_iter = 500;
step_size = 0.00001;
batch_size = 500;
for i = 1: max_iter * num_train / batch_size
    r = mod(i-1, num_train / batch_size);
    samp = X_train(r * batch_size + 1 : (r+1) * batch_size, :);
    samp_y = y_train(r * batch_size + 1 : (r+1) * batch_size, :);
    p = exp(-(samp * w));
    h = p ./ repmat(sum(p, 2), 1, num_class);
    w_prev = w;
    w = w - step_size * samp' * (samp_y - h);
    if norm(w_prev - w)/norm(w) < 0.001
        break;
    end
    if (mod (i - 1, 1000) == 0)
        fprintf('iteration %d, diff = %f\n', i, norm(w_prev - w)/norm(w));
    end
end

p = exp(-(x_test * w));
h = p ./ repmat(sum(p, 2), 1, num_class);
[~, y_test] = max(h, [], 2);
y_test = y_test - 1;





