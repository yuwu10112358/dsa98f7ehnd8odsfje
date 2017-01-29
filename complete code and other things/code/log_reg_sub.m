%% Train the substitute Logistic Regression model and obtain the parameters

clc
close all

% load the MNIST data
images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');


% Create Logistic Regression Model Substitute
% initialize the parameters as in Papernot et al. (2016)
lambda = 0.1;
tau = 1;
sigma = 3;
base_size = 100;
k = base_size * 4;

% Initialize the training data for the oracles (first 50,000 samples)
X = images(:,1:50000);
% X = [ones(1,size(X,2)); X];
y = labels(1:50000);

% initialize the training data for the substitute model - first 100 samples
% from the test set (unseen by the oracle)
S = images(:,57001:(57000 + base_size))';
[W_oracle, b_oracle] = LR_Train_Oracle(X, y, num_class);
%S = [ones(size(S,1),1) S];
% ORACLE LABEL - KNN
y_S = LR_predict(S', W_oracle, b_oracle);

% Implement logistic regression periodical step size (PSS) and reservoir
% sampling (RS) training set augmentation
max_rho = 10;
error_lr = zeros(max_rho, 1);

for rho=1:max_rho
    % implement PSS
    lambda_rho = lambda*(-1)^floor(rho/tau);
    
    % initialize the Jacobian matrix
    J = zeros(size(S,2),max(y_S)-min(y_S)+1);
    
    % obtain the parameters for logistic regression substitute model
    [W, b] = LR_Train_Oracle(S',y_S,10);
    
   [error_lr(rho), error_knn, I_lr, I_knn] = test(X,y,W, b, images,labels);
   fprintf('Interation %d: error_lr: %f, error_knn: %f\n', rho, error_lr(rho), error_knn);
%     I_lr
%     I_knn
    % if this is not the last iteration, augment the training set
    if rho < 9
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
                y_S = [y_S; LR_predict(new', W_oracle, b_oracle)];
            end
        else
            S = augment_rs(S,y_S,k,J,lambda_rho);
            % augment the appropriate labels
            y_S = [y_S; LR_predict(S(end-k+1:end,:)', W_oracle, b_oracle)];
        end
    end
end
plot(0:(max_rho-1), error_lr);
