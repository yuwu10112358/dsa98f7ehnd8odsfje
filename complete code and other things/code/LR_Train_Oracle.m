function [W, b] = LR_Train_Oracle (X_train, y_train, num_class)

dim_train = size(X_train);

image_s = dim_train(1);
num_train = dim_train(2);
X_train = [X_train; ones(1, num_train)]';
y_train = onehot(y_train, num_class);
w = zeros(image_s + 1, num_class);

if num_train < 1000 
    step_size = 0.0001;
    max_iter = 100;
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
    end
else
    max_iter = 500;
    step_size = 0.0001;
    batch_size = 100;
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
    end
end

W = w(1:image_s, :);
b = w(image_s + 1, :)';









