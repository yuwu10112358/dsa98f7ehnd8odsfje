function [w] = LR_train (X_train, y_train, num_class)

dim_train = size(X_train);

image_s = dim_train(2);
num_train = dim_train(2);
X_train = [X_train; ones(1, num_train)]';
y_train = onehot(y_train, num_class);
w = zeros(image_s, num_class);

max_iter = 500;
step_size = 0.0001;
for i = 1: max_iter
    samp = X_train;
    samp_y = y_train;
    p = exp(-(samp * w));
    h = p ./ repmat(sum(p, 2), 1, num_class);
    w_prev = w;
    w = w - step_size * samp' * (samp_y - h);
    if norm(w_prev - w)/norm(w) < 0.001
        break;
    end
    if (mod (i - 1, 100) == 0)
        fprintf('iteration %d, diff = %f\n', i, norm(w_prev - w)/norm(w));
    end
end
