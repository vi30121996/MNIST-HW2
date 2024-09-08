% Load the data from the .mat file
load('data21.mat');

% Initialize the image matrix
imgs = zeros(28*10, 28*10);

% For each row
for i = 1:10
    % For each column
    for j = 1:10
        % Generate a random Z vector
        Z = randn(10, 1);
        
        % Calculate W1, Z1, W2 and X
        W1 = A_1 * Z + B_1;
        Z1 = max(W1, 0); % Relu
        W2 = A_2 * Z1 + B_2;
        X = 1 ./ (1 + exp(W2)); % Sigmoid

        % Reshape X into a 28x28 image
        img = reshape(X, 28, 28);
        
        % Place the image in the corresponding location in the imgs matrix
        imgs((i-1)*28+1:i*28, (j-1)*28+1:j*28) = img;
    end
end

% Display the images
imshow(imgs, []);
