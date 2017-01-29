function [ S_new ] = augment_rs( S_old,y_S,k,J,lambda )
%AUGMENT Implement the Jacobian-based augmentation with Reservoir Sampling
%as in Papernot et al. (2016)
%   Input:
%   S_old - previous iteration training set (m_old x n)
%   k - the number of randomly selected samples for reservoir sampling
%   J - Jacobian for the previously trained classifier
%   lambda - parameter determining the augmentation step-size
%   
%   Output:
%   S_new - augmented training set

% size of the previous training set
m_old = size(S_old,1);
% initialize the new training set
S_new = zeros(m_old+k,size(S_old,2));
% the first m_old elements of S_new are S_old
S_new(1:m_old,:) = S_old;

% for the first k new samples populate the augmented set
for i=1:k
    S_new(m_old+i,:) = S_old(i,:) + lambda*sign(J(:,y_S(i)+1,i)');
end

% randomly update some of the new k samples utilizing the new samples
% utilized above
for i=(k+1):m_old
   % generate a random integer
   r = randi([1,i],1); 
   % check if the integer is within the new k samples
   if r <= k
       S_new(m_old+r,:) = S_old(i,:) + lambda*sign(J(:,y_S(i)+1,i)');
   end
end

end
