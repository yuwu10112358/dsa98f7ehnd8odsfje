% W_sub_LR = LR_Train_Oracle(X_sub_LR, label_sub_LR, num_class);
% adx_LR = Adversarial_LR(x_test, y_test, W_sub_LR);
% labels_adx_LR = LR_predict(adx_LR, W_oracle, b_oracle);
% fprintf('the missclassification rate for LR sub LR is %f\n', sum(labels_adx_LR ~= y_test)/length(y_test) * 100);
% 
% W_sub_SVM = LR_Train_Oracle(X_sub_SVM, label_sub_SVM, num_class);
% adx_SVM = Adversarial_LR(x_test, y_test, W_sub_SVM);
% labels_adx_SVM = SVM_predict(model_arr, adx_SVM, 10);
% fprintf('the missclassification rate for LR sub SVM is %f\n', sum(labels_adx_SVM ~= y_test)/length(y_test) * 100);

% W_sub_kNN = LR_Train_Oracle(X_sub_kNN, label_sub_kNN, num_class);
% adx_kNN = Adversarial_LR(x_test, y_test, W_sub_kNN);
% labels_adx_kNN = kNN_slow(X_train, y_train, adx_kNN);
% fprintf('the missclassification rate for LR sub LR is %f\n', sum(labels_adx_kNN ~= y_test)/length(y_test) * 100);