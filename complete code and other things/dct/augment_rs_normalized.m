function [ S_new ] = augment_rs_normalized( S_old,y_S,k,J,lambda,norm_X, average, var)

m_old = size(S_old,1);
% initialize the new training set
S_new = zeros(m_old+k,size(S_old,2));
% the first m_old elements of S_new are S_old
S_new(1:m_old,:) = S_old;

var_temp = var(var ~= 0);
average_temp = average(var ~= 0);
    
% for the first k new samples populate the augmented set
for i=1:k
    norm_new = norm_X(:, i) + lambda*sign(J(:,y_S(i)+1,i));
    un_nomalized_new = zeros(size(S_old, 2), 1);
    un_nomalized_new(var ~= 0) = sqrt(var_temp) .* norm_new + average_temp;
    un_nomalized_new(var == 0) = average(var == 0);
    S_new(m_old+i,:) = un_nomalized_new';
end

for i=(k+1):m_old
   % generate a random integer
   r = randi([1,i],1); 
   % check if the integer is within the new k samples
   if r <= k
       norm_new = norm_X(:, i) + lambda*sign(J(:,y_S(i)+1,i));
        un_nomalized_new = zeros(size(S_old, 2), 1);
        un_nomalized_new(var ~= 0) = sqrt(var_temp) .* norm_new + average_temp;
        un_nomalized_new(var == 0) = average(var == 0);
       S_new(m_old+r,:) = un_nomalized_new';
   end
end

end
