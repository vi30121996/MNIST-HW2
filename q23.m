% Load the data from the .mat file
load('data21.mat');
load('data23.mat');

% Number of iterations for the gradient descent
num_iterations = 1000;

% Initialize the Z vector
Z = randn(10, 1);

% Initialize the matrix for storing cost function values
num_noisy_images = size(X_n, 2);
M = size(X_n, 1);
J_values_all = zeros(num_iterations, num_noisy_images);

% Initialize the transformation matrix T
lowres_size = 7;
T = zeros(lowres_size^2, 28*28);
for i = 1:lowres_size^2
    highres_row = 4 * floor((i - 1) / lowres_size);
    highres_col = 4 * mod(i - 1, lowres_size);
    for j = 0:3
        for k = 0:3
            highres_index = (highres_row + j) * 28 + (highres_col + k) + 1;
            T(i, highres_index) = 1/16;
        end
    end
end

% Initialize the figure for displaying reconstructed 8s
figure;

% For each image
for img_id = 1:num_noisy_images
    % Extract the non-lost part of the image
    non_lost_part = X_n(:, img_id);
    
    % For each iteration
    for i = 1:num_iterations
        % Calculate W1, Z1, W2, and X
        W1 = A_1 * Z + B_1;
        Z1 = max(W1, 0); % ReLU activation
        W2 = A_2 * Z1 + B_2;
        X = 1 ./ (1 + exp(W2)); % Sigmoid activation
        
        % Compute the gradient of Φ(X)
        U2 = (1 / norm(T * X - non_lost_part)^2) * (2 * T' * (T * X - non_lost_part));
        % Compute the gradient of sigmoid
        F2 = (-exp(W2) ./ (1 + exp(W2)).^2)';
        % Compute the gradient of ReLU
        F1 = double(W1 > 0);
        % Compute V2 and U1
        V2 = U2 .* F2;
        U1 = A_2' * V2;
        % Compute V1 and U0
        V1 = U1 .* F1;
        U0 = A_1' * V1;

        % Compute the total gradient
        gradient = M * U0 + 2 * Z;
        
        % Update Z using Adam optimizer
        learning_rate = 0.01;  
        b1 = 0.9;
        b2 = 0.999;
        epsilon = 1e-8;
        m = zeros(size(Z));
        v = zeros(size(Z));
        m = b1 * m + (1 - b1) * gradient;
        v = b2 * v + (1 - b2) * (gradient.^2);
        m_hat = m / (1 - b1^i);
        v_hat = v / (1 - b2^i);
        Z = Z - learning_rate * m_hat ./ (sqrt(v_hat) + epsilon);
        Z = mean(Z, 2);  % Average over the 2nd dimension to ensure Z stays as a 10x1 vector

        % Calculate the cost function J(Z) and store in J_values_all
        J = M * log(norm(T * X - non_lost_part)^2) + norm(Z)^2;
        J_values_all(i, img_id) = J;
    end
    
    % Generate the image from Z
    W1 = A_1 * Z + B_1;
    Z1 = max(W1, 0); % ReLU activation
    W2 = A_2 * Z1 + B_2;
    X_recon = 1 ./ (1 + exp(W2)); % Sigmoid activation

    % Reshape the reconstructed X, the noisy image, and the ideal image into 28x28 images
    img_recon = reshape(X_recon, 28, 28);
    img_noisy = reshape(X_n(:, img_id), lowres_size, lowres_size);
    img_ideal = reshape(X_i(:, img_id), 28, 28);

    % Plot the ideal, noisy, and reconstructed images side by side
    subplot(num_noisy_images, 3, (img_id-1)*3 + 1);
    imshow(img_ideal, []);
    title('Ideal Image');
    subplot(num_noisy_images, 3, (img_id-1)*3 + 2);
    imshow(img_noisy, []);
    title('Noisy Image');
    subplot(num_noisy_images, 3, (img_id-1)*3 + 3);
    imshow(img_recon, []);
    title('Reconstructed Image');
end

% Set the figure title
sgtitle('Reconstructed 8s');

% Plot cost function values
figure;
plot(1:num_iterations, J_values_all);
xlabel('Iteration');
ylabel('Cost function value');
legend('Image 1', 'Image 2', 'Image 3', 'Image 4');
title('Cost function values for each image');
