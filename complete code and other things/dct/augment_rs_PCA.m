function [ S_new ] = augment_rs_PCA( S_old,y_S,k,J,lambda,X_new, T)

m_old = size(S_old,1);
% initialize the new training set
S_new = zeros(m_old+k,size(S_old,2));
% the first m_old elements of S_new are S_old
S_new(1:m_old,:) = S_old;
    
% for the first k new samples populate the augmented set
for i=1:k
    norm_new = T * (X_new(:, i) + lambda*sign(J(:,y_S(i)+1,i)));
    un_nomalized_new = norm_new;
    S_new(m_old+i,:) = un_nomalized_new';
end

for i=(k+1):m_old
   % generate a random integer
   r = randi([1,i],1); 
   % check if the integer is within the new k samples
   if r <= k
       norm_new = T * (X_new(:, i) + lambda*sign(J(:,y_S(i)+1,i)));
       un_nomalized_new = norm_new;
       S_new(m_old+r,:) = un_nomalized_new';
   end
end

end
