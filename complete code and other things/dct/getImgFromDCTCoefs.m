function [img] = getImgFromDCTCoefs(coefs, w, h, w_factor, h_factor)
% image is a column vector of pixel values, w, h, is the width and height
% of the original image, w_factor  and h_factor are the compression factor
% in their respective dimension

img_coefs = zeros(h, w);
img_coefs(1:(h/h_factor),1:(w/w_factor)) = reshape(coefs, (h/h_factor), (w/w_factor));
img = reshape(ifft2(img_coefs), [], 1);
