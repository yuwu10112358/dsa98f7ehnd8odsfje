function [X, labels, sizes] = generate_LR_sub_Dataset_for_LR_with_PCA(W_oracle, b_oracle, init_X, factor, max_iter)

%X are the generated dataset from which sub models can be developed; dim is 784 X n
%labels are the LR oracle labels for X; dim is n x 1
% sizes is a k x 1 vector; k is the max number of iterations to generate X,
% the value of each element means that for the i_th iteration,  the dataset 
% generated is X(:, 1:sizes(i))

lambda = 0.1;
tau = 1;
sigma = 3;
base_size = size(init_X, 2);
k = base_size * 4;

S = init_X';
y_S = LR_predict(S', W_oracle, b_oracle);

max_rho = max_iter;
sizes = zeros(max_rho, 1);
coefs = 0;
disp('generating datasets for LR substitute');
for rho=1:max_rho
    fprintf('Iteration %d\n', rho);
    sizes(rho) = size(S, 1);
    % implement PSS
    lambda_rho = lambda*(-1)^floor(rho/tau);
    norm_X = S';
    m1 = size(norm_X, 1);
    m2 = size(norm_X, 2);
    if (m1 > m2)
        T = eye(size(norm_X, 1));
    else
        [coefs, ~, ~, ~, explained] = pca(S);
        T = coefs(:,1:floor(m1/factor));
    end
    X_new = T'*S';
    J = zeros(size(X_new, 1), size(W_oracle, 2), size(S, 1));
    % obtain the parameters for logistic regression substitute model
    
    [W, b] = LR_Train_Oracle(X_new,y_S,10);

    % if this is not the last iteration, augment the training set
    if rho < max_rho
        %J is 784 x 10 x l
        for l = 1:size(S, 1)
            J(:, :, l) = 1 / sum(exp(-W' * X_new(:, l)))^2 * ...
                (sum(exp(-W' * X_new(:, l))) * -repmat(exp(-W' * X_new(:, l))', size(X_new, 1), 1) .* W + ...
                repmat(exp(-W' * X_new(:, l))', size(X_new, 1), 1) .* repmat(W * exp(-W' * X_new(:, l)), 1, size(W, 2)));
        end

        if rho <= sigma
            for i=1:size(S,1)
                % oracle output of ith training sample
                O = y_S(i);
                norm_new = T * (X_new(:, i) + lambda_rho*sign(J(:,O+1,i)));
                un_nomalized_new = norm_new;
                S = [S; un_nomalized_new'];
                y_S = [y_S; LR_predict(un_nomalized_new, W_oracle, b_oracle)];
            end
        else
            S = augment_rs_PCA(S,y_S,k,J,lambda_rho, X_new, T);
            % augment the appropriate labels
            
            y_S = [y_S; LR_predict(S(end-k+1:end,:)', W_oracle, b_oracle)];
        end
    end
end

X = S';
labels = y_S;
sizes(end) = size(S, 1);








