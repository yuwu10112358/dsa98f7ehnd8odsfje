function[adv_x] = Adversarial_LR(x, true_labels, W)
%x is 784 x n
%labels are n x 1
%W is 784 x 10
num_samp = size(x, 2);
l = size(x, 1);
adv_x = zeros(l, num_samp);
eps = 0.3;
for i = 1:num_samp
    c = true_labels(i) + 1;
    h = exp(-W' * x(:, i));
    delta = (-W(:, c) * h(c) * sum(h) + h(c) * W * h) / (sum(h))^2;
    %because image is black & white, use the gradient to determine which
    %pixel to change
    adv_x(:, i) = x(:, i) - eps * sign(delta);
%     temp = direction;
%     temp(direction < 0) = 0;
%     change_ind = xor(temp, x(:, i)); %1 indicate pixel can be changed, 0 not
%     ind = 1:l;
%     ind = ind(change_ind);
%     num_pix_change = min(floor(eps * l), sum(change_ind));
%     [~, max_i] = sort(abs(delta(ind)));
%     change = zeros(l, 1);
%     change(max_i(1:num_pix_change)) = sign(delta(max_i(1:num_pix_change)));
%     adv_x(:, i) = x(:, i) - change;
end

