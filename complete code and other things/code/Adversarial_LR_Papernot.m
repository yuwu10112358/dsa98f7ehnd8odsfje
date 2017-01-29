function[adv_x] = Adversarial_LR_Papernot(x, true_labels, W)
%x is 784 x n
%labels are n x 1
%J is 784 x 10

%784 x 10 x 10000 (size(x,2))
num_samp = size(x, 2);
num_features = size(x,1);
adv_x = zeros(size(x));
gamma = floor(0.1 * num_features);
eps = 1;

% build the jacobian
for l = 1:size(x,2)
    J(:, :, l) = 1 / sum(exp(-W' * x(:, l)))^2 * ...
                    (sum(exp(-W' * x(:, l))) * -repmat(exp(-W' * x(:, l)), 1, size(x, 1))' .* W + ...
                    repmat(exp(-W' * x(:, l)), 1, size(x, 1))' .* repmat(W * exp(-W' * x(:,l)), 1, size(W, 2)));
end

for l=1:num_samp
   % find the labels that are not equal to the true class of t

   index_not = ((true_labels(l) + 1) ~= 1:10);
   index = ((true_labels(l) + 1) == 1:10);
   % calculate the saliency values
    S_p = J(:,logical(index),l) .* (J(:,logical(index),l) > 0) .* (sum(J(:,logical(index_not),l), 2) < 0).* abs(sum(J(:,logical(index_not),l),2));
    S_m = J(:,logical(index),l) .* (J(:,logical(index),l) < 0) .* (sum(J(:,logical(index_not),l), 2) > 0).* abs(sum(J(:,logical(index_not),l),2));
    % reorder the rows of the saliency matrix from largest to smallest 
    [S, dir] = max([S_m, S_p], [],2);
    dir = 2 * dir - 3;
    [~, ind] = sortrows(S,-1);
    delta = x(ind(1:gamma),l);
    adv_x(:,l) = x(:,l);
    adv_x(ind(1:gamma), l) = adv_x(ind(1:gamma), l) - eps * (delta);
end
