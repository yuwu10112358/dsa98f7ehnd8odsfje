%% Test whether the 'log_reg_sub.m' model performs with similar accuracy to
% 'knn_oracle.m' model
function [success_rate_lr,success_rate_knn, I_lr, I_knn] = test(,y,W, b, images,labels)

X_test = images(:,end-10000+1:end);
y_test = labels(end-10000+1:end);
success_rate_knn = 0;
I_knn = 0;
y_lr_sub = LR_predict(X_test,W, b);
y_lr_oracle = LR_predict(X_test, W_oracle, b_oracle);
%y_knn_oracle = knn_slow(X,y,X_test');

%fprintf('Success rate of substitute model: \n')
success_rate_lr = sum(y_lr_oracle == y_lr_sub)/length(y_lr_sub)*100;
%success_rate_knn = sum(y_knn_oracle == y_lr_sub)/length(y_lr_sub)*100;

% find the indeces of the correct numbers
[I_lr] = find(y_lr_oracle == y_lr_sub);
%[I_knn] = find(y_knn_oracle == y_lr_sub);

% fprintf('Success rate of oracle model: \n')
% success_rate_lr = sum(y_lr_oracle == y_test)/length(y_test)*100;
% success_rate_knn = sum(y_knn_oracle == y_test)/length(y_test)*100;

end