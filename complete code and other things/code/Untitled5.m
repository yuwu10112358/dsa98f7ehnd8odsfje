images_train = loadMNISTImages('../data/train-images.idx3-ubyte');
labels_train = loadMNISTLabels('../data/train-labels.idx1-ubyte');
images_test = loadMNISTImages('../data/t10k-images.idx3-ubyte');
labels_test = loadMNISTLabels('../data/t10k-labels.idx1-ubyte');

img1 = images_train(:, 1, :);

test1 = fft(img1);
figure;
imshow(reshape(img1, 28, 28));
figure;
imshow(reshape(test1, 28, 28))
test2 = ifft(test1);
figure;
imshow(reshape(test2, 28, 28));