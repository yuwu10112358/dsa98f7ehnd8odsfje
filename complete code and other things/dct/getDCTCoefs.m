function [coef] = getDCTCoefs(image, w, h, w_factor, h_factor)
% image is a column vector of pixel values, w, h, is the width and height
% of the original image, w_factor  and h_factor are the compression factor
% in their respective dimension

t = fft2(reshape(image, h, w));
coef = reshape(t(1:(h/h_factor), 1:(w/w_factor)), [], 1);
