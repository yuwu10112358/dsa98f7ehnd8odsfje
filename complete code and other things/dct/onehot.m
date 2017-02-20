function [converted_labels] = onehot (labels, num_class)
converted_labels = zeros(length(labels), num_class);
for i = 1:length(labels)
    converted_labels(i, labels(i) + 1) = 1;
end