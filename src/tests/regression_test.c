#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../config/config.h"
#include "../network/network.h"
#include "../utils/math_utils.h"
#include "../utils/utils.h"
#include "../utils/random_utils.h"
#include <string.h>
#include "../optimizer/optimizer.h"

// Synthetic regression data generator.
// Generates a dataset where each sample is 100-dimensional.
// The first feature is uniformly sampled in [0, 2*pi] and used to compute y = sin(x) + noise.
// The remaining 99 features are set to 0.
void generate_regression_data(int batch_size, Matrix **input, Matrix **labels) {
    int input_dim = 100;  // Set input dimension to 100.
    *input = create_matrix(batch_size, input_dim);
    *labels = create_matrix(batch_size, 1);  // Single output per sample.
    
    // Generate linearly spaced values for the first feature in [0, 2*pi].
    //double interval = (2.0 * M_PI) / (batch_size - 1);
    
    for (int i = 0; i < batch_size; i++) {
        double x = (rand() % 100) + 1;
        // Set the first feature.
        (*input)->data[i * input_dim + 0] = x;
        // Fill the remaining 99 features with zeros.
        for (int j = 1; j < input_dim; j++) {
            (*input)->data[i * input_dim + j] = 0.0;
        }
        // Compute the target using the first feature.
        double noise = random_gaussian(0.0, 0.1);
        (*labels)->data[i] = (log10(x+1) + noise)*10;
        printf("%f, ", (*labels)->data[i]);
        fflush(stdout);
    }
}

// Compute Mean Squared Error loss between prediction and target matrices.
double compute_mse_loss(const Matrix *pred, const Matrix *target) {
    double sum = 0.0;
    int total = pred->rows * pred->cols;
    for (int i = 0; i < total; i++) {
        double diff = pred->data[i] - target->data[i];
        sum += diff * diff;
    }
    return sum / total;
}

// Compute the gradient of the MSE loss with respect to predictions.
// For MSE L = (1/total)*sum((pred - target)^2),
// the derivative is (2/total)*(pred - target).
Matrix* compute_mse_loss_gradient(const Matrix *pred, const Matrix *target) {
    Matrix *grad = copy_matrix(pred);
    int total = pred->rows * pred->cols;
    

    for (int i = 0; i < total; i++) {
        
        grad->data[i] = (2.0 / total) * (pred->data[i] - target->data[i]);
    }
    
    return grad;
}

int main(void) {
    // ------------------------------
    // Setup configuration and network.
    // ------------------------------
    Config cfg;
    init_config(&cfg);
    // Override configuration for testing.
    cfg.num_layers = 4;
    // Use four logical layers with desired neuron outputs.
    strncpy(cfg.neurons_per_layer, "64,128,64,1", sizeof(cfg.neurons_per_layer) - 1);
    cfg.neurons_per_layer[sizeof(cfg.neurons_per_layer) - 1] = '\0';
    // Specify layer types. For this test, use "linear,linear,stochastic,linear".
    strncpy(cfg.layer_types, "linear,linear,linear,linear", sizeof(cfg.layer_types) - 1);
    cfg.layer_types[sizeof(cfg.layer_types) - 1] = '\0';
    // Set Prior and Posterior methods.
    cfg.prior_type = 0;         // Use Laplace prior.
    cfg.prior_variance = .1;
    cfg.posterior_method = 2;   // Use Flipout posterior.
    // Dropout probability (used by dropout layers).
    cfg.dropout_prob = 0.5;
    cfg.learning_rate = .00000005;
    cfg.kl_weight = 1;
    
    printf("Creating network with %d logical layers\n", cfg.num_layers);
    printf("Neurons per layer: %s\n", cfg.neurons_per_layer);
    printf("Layer types: %s\n", cfg.layer_types);
    printf("Prior type: %d, Posterior method: %d\n", cfg.prior_type, cfg.posterior_method);
    
    Network *net = create_network(&cfg);
    if (!net) {
        handle_error("Failed to create network.");
    }
    printf("Network created with %d internal layers (including projection layers).\n", net->num_layers);
    
    // ------------------------------
    // Generate synthetic regression data.
    // ------------------------------
    int batch_size = 100;
    Matrix *X, *Y;
    generate_regression_data(batch_size, &X, &Y);
    
    // ------------------------------
    // Simulate a training loop with backpropagation.
    // ------------------------------
    int epochs = 25;
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Forward pass: compute predictions.
        Matrix *pred = network_forward(net, X, 1); // stochastic forward pass
        
        // Compute loss components.
        double mse = compute_mse_loss(pred, Y);
        double kl = network_total_kl(net);
        double total_loss = mse + cfg.kl_weight * kl;
        printf("Epoch %d: MSE = %f, KL = %f, Total Loss = %f\n", epoch, mse, kl, total_loss);
        //printf("0;jh;lhj;");
        fflush(stdout);

        // Compute gradient of the MSE loss with respect to predictions.
       // printf("DEBUG: pred dimensions: %d x %d, data pointer: %p\n", pred->rows, pred->cols, (void*)pred->data);
        fflush(stdout);

        Matrix *grad_loss = compute_mse_loss_gradient(pred, Y);
      //  printf("1");
        fflush(stdout);

        
        // Backward pass: propagate gradients through the network.
        Matrix *grad_input = network_backward(net, grad_loss, &cfg);
       // printf("2");
        fflush(stdout);

        // Free intermediate gradient matrices.
        free_matrix(grad_loss);
        free_matrix(grad_input);
        
        // Update parameters using the optimizer.
        network_update_params(net, cfg.learning_rate);
      //  printf("3");
        fflush(stdout);

        
        free_matrix(pred);
    }
    
    // ------------------------------
    // Evaluate uncertainty via Monte Carlo sampling.
    // ------------------------------
    int num_samples = 50;
    Matrix **pred_samples = malloc(sizeof(Matrix*) * num_samples);
    for (int i = 0; i < num_samples; i++) {
        pred_samples[i] = network_forward(net, X, 1); // stochastic mode
    }
    
    // Compute per-sample mean and variance.
    Matrix *mean_pred = create_matrix(batch_size, 1);
    Matrix *var_pred = create_matrix(batch_size, 1);
    for (int b = 0; b < batch_size; b++) {
        double sum = 0.0;
        for (int i = 0; i < num_samples; i++) {
            sum += pred_samples[i]->data[b];
        }
        double mean = sum / num_samples;
        mean_pred->data[b] = mean;
        
        double sq_sum = 0.0;
        for (int i = 0; i < num_samples; i++) {
            double diff = pred_samples[i]->data[b] - mean;
            sq_sum += diff * diff;
        }
        var_pred->data[b] = sq_sum / num_samples;
    }
    
    printf("Uncertainty estimates (first 5 samples):\n");
    for (int b = 0; b < 5; b++) {
        printf("Sample %d: Mean = %f, Variance = %f\n", b, mean_pred->data[b], var_pred->data[b]);
    }
    
    // Clean up.
    for (int i = 0; i < num_samples; i++) {
        free_matrix(pred_samples[i]);
    }
    free(pred_samples);
    free_matrix(mean_pred);
    free_matrix(var_pred);
    free_matrix(X);
    free_matrix(Y);
    free_network(net);
    
    printf("Regression test completed successfully.\n");
    return 0;
}
