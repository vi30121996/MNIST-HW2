% Load the data
load('data21.mat');
load('data22.mat');

% Define the number of iterations and N_values
num_iterations = 1000;
N_values = [500, 400, 350, 300];

% Initialize the cost values vector with dimensions [num_iterations, 4, numel(N_values)]
J_values_all = zeros(num_iterations, 4, numel(N_values));

% Initialize the images arrays
damaged_imgs = zeros(28, 28, 4, numel(N_values));
reconstructed_imgs = zeros(28, 28, 4, numel(N_values));

% Iterate over different ideal images and N values
for img_id = 1:4
    for N_id = 1:numel(N_values)
        N = N_values(N_id);
        % Get the ideal image, the noisy image, and the non-lost part
        X_i_col = X_i(:, img_id);
        X_n_col = X_n(:, img_id);
        non_lost_part = X_n_col(1:N_values(N_id), :);

        % Initialize Z
        Z = randn(10, 1);

        % Define T
        T = [eye(784); zeros(N_values(N_id)-784, 784)];
        T = T(1:N_values(N_id), :);  % Select the first N rows

        % Create a damaged version of the image with missing pixels
        damaged_img = X_n_col;
        damaged_img(N_values(N_id)+1:end) = 0;  % Set the lost pixels to black
        damaged_imgs(:, :, img_id, N_id) = rot90(reshape(damaged_img, 28, 28)', -1);

        % Perform inpainting
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
            U1 = mean(A_2' * V2, 2);  
            % Compute V1 and U0
            V1 = U1 .* F1;
            U0 = mean(A_1' * V1, 2);  
            
            % Compute the total gradient
            gradient = N * U0 + 2 * Z;
            
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
            J = N * log(norm(T * X - non_lost_part)^2) + norm(Z)^2;
            J_values_all(i, img_id, N_id) = J;

            % Generate the reconstructed image
            X_reconstructed = 1 ./ (1 + exp(-W2)); % Sigmoid activation
            reconstructed_imgs(:, :, img_id, N_id) = rot90(reshape(X_reconstructed, 28, 28)', -1);
        end
    end
end

% Define the four different 8 images
ideal_imgs = cell(1, 4);
for img_id = 1:4
    ideal_imgs{img_id} = rot90(reshape(X_i(:, img_id), 28, 28)', -1);
end

% Display the images in separate figures
for img_id = 1:4
    figure;
    
    for N_id = 1:numel(N_values)
        % Display the ideal image
        subplot(numel(N_values), 3, (N_id-1)*3 + 1);
        imshow(ideal_imgs{img_id});
        if N_id == 1
            title('Ideal Image');
        end
        
        % Display the damaged image
        subplot(numel(N_values), 3, (N_id-1)*3 + 2);
        imshow(damaged_imgs(:, :, img_id, N_id));
        if N_id == 1
            title('Damaged Image');
        end
        ylabel(['N = ', num2str(N_values(N_id))]);

        % Display the reconstructed image
        subplot(numel(N_values), 3, (N_id-1)*3 + 3);
        imshow(reconstructed_imgs(:, :, img_id, N_id));
        if N_id == 1
            title('Reconstructed Image');
        end
    end
end

% Display the cost evolution in separate figures
for img_id = 1:4
    figure;
    hold on;
    for N_id = 1:numel(N_values)
        plot(1:num_iterations, squeeze(J_values_all(:, img_id, N_id)), 'DisplayName', ['N = ', num2str(N_values(N_id))]);
    end
    xlabel('Iteration');
    ylabel('Cost');
    title(['Cost Evolution for Image ', num2str(img_id)]);
    legend;
    hold off;
end
