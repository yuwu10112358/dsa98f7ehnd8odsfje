function [X, labels, sizes] = generate_LR_sub_Dataset_for_LR_with_normalization(W_oracle, b_oracle, init_X)

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

max_rho = 10;
sizes = zeros(max_rho, 1);

disp('generating datasets for LR substitute');
for rho=1:max_rho
    fprintf('Iteration %d\n', rho);
    sizes(rho) = size(S, 1);
    % implement PSS
    lambda_rho = lambda*(-1)^floor(rho/tau);
    %normalization
    un_normalized_X = S';
    average = 1/size(un_normalized_X,2)*sum(un_normalized_X,2);
    var = 1/size(un_normalized_X,2)* sum((un_normalized_X - repmat(average, 1, size(un_normalized_X,2))).^2, 2);
    X_temp = un_normalized_X(var ~= 0, :);
    var_temp = var(var ~= 0);
    average_temp = average(var ~= 0);
    norm_X = (X_temp - repmat(average_temp, 1, size(un_normalized_X,2))) ./repmat(sqrt(var_temp), 1, size(un_normalized_X,2));
    
    % obtain the parameters for logistic regression substitute model
    
    [W, b] = LR_Train_Oracle(norm_X,y_S,10);

    % if this is not the last iteration, augment the training set
    if rho < max_rho
        %J is 785 x 10 x l
        for l = 1:size(S, 1)
            J(:, :, l) = 1 / sum(exp(-W' * norm_X(:, l)))^2 * ...
                (sum(exp(-W' * norm_X(:, l))) * -repmat(exp(-W' * norm_X(:, l))', size(norm_X, 1), 1) .* W + ...
                repmat(exp(-W' * norm_X(:, l))', size(norm_X, 1), 1) .* repmat(W * exp(-W' * norm_X(:, l)), 1, size(W, 2)));
        end

        if rho <= sigma
            for i=1:size(S,1)
                % oracle output of ith training sample
                O = y_S(i);
                norm_new = norm_X(:, i) + lambda_rho*sign(J(:,O+1,i));
                un_nomalized_new = zeros(size(S, 2), 1);
                un_nomalized_new(var ~= 0) = sqrt(var_temp) .* norm_new + average_temp;
                un_nomalized_new(var == 0) = average(var == 0);
                % augment the training set and labels
                S = [S; un_nomalized_new'];
                y_S = [y_S; LR_predict(un_nomalized_new, W_oracle, b_oracle)];
            end
        else
            S = augment_rs_normalized(S,y_S,k,J,lambda_rho, norm_X, average, var);
            % augment the appropriate labels
            
            y_S = [y_S; LR_predict(S(end-k+1:end,:)', W_oracle, b_oracle)];
        end
    end
end

X = S';
labels = y_S;








