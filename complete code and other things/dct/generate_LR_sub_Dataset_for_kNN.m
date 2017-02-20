function [X, labels, sizes] = generate_LR_sub_Dataset_for_kNN(X_train, y_train, init_X, max_iter)

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
y_S = knn_oracle(X_train, y_train, S');

max_rho = max_iter;
sizes = zeros(max_rho, 1);
disp('generating datasets for kNN substitute');
for rho=1:max_rho
    fprintf('Iteration %d\n', rho);
    sizes(rho) = size(S, 1);
    % implement PSS
    lambda_rho = lambda*(-1)^floor(rho/tau);
        
    % obtain the parameters for logistic regression substitute model
    [W, b] = LR_Train_Oracle(S',y_S,10);

    % if this is not the last iteration, augment the training set
    if rho < max_rho
        %J is 785 x 10 x l
        for l = 1:size(S, 1)
            J(:, :, l) = 1 / sum(exp(-W' * S(l, :)'))^2 * ...
                (sum(exp(-W' * S(l, :)')) * -repmat(exp(-W' * S(l, :)')', size(S, 2), 1) .* W + ...
                repmat(exp(-W' * S(l, :)')', size(S, 2), 1) .* repmat(W * exp(-W' * S(l, :)'), 1, size(W, 2)));
        end

        if rho <= sigma
            for i=1:size(S,1)
                % oracle output of ith training sample
                O = y_S(i);
                new = S(i,:) + lambda_rho*sign(J(:,O+1,i)');
                % augment the training set and labels
                S = [S; new];
                y_S = [y_S; knn_oracle(X_train, y_train, new')];
            end
        else
            S = augment_rs(S,y_S,k,J,lambda_rho);
            % augment the appropriate labels
            y_S = [y_S; knn_oracle(X_train, y_train, S(end-k+1:end,:)')];
        end
    end
end

X = S';
labels = y_S;
sizes(end) = size(S, 1);