#include "bayesian_linear.h"
#include "../utils/utils.h"          // For handle_error() and logging.
#include "../utils/random_utils.h"   // For random number generation.
#include <stdlib.h>
#include <math.h>
#include "../config/config.h"

// Create a new Bayesian linear layer.
BayesianLinear* create_bayesian_linear(int input_dim, int output_dim) {
    BayesianLinear *layer = (BayesianLinear*)malloc(sizeof(BayesianLinear));
    if (!layer) {
        handle_error("Failed to allocate memory for BayesianLinear layer.");
    }
    layer->input_dim = input_dim;
    layer->output_dim = output_dim;
    
    // Allocate matrices for weights.
    layer->W_mean = create_matrix(output_dim, input_dim);
    layer->W_logvar = create_matrix(output_dim, input_dim);
    if (!layer->W_mean || !layer->W_logvar) {
        handle_error("Failed to allocate matrices for weights.");
    }
    
    // Allocate arrays for biases.
    layer->b_mean = (double*)malloc(sizeof(double) * output_dim);
    layer->b_logvar = (double*)malloc(sizeof(double) * output_dim);
    if (!layer->b_mean || !layer->b_logvar) {
        handle_error("Failed to allocate arrays for biases.");
    }
    
    // Initialize weights and biases.
    for (int i = 0; i < output_dim; i++) {
        for (int j = 0; j < input_dim; j++) {
            int idx = i * input_dim + j;
            layer->W_mean->data[idx] = random_gaussian(0.0, 0.1);
            layer->W_logvar->data[idx] = -5.0;
        }
        layer->b_mean[i] = random_gaussian(0.0, 0.1);
        layer->b_logvar[i] = -5.0;
    }
    
    // Allocate gradient matrices and arrays:
    layer->dW_mean = create_matrix(output_dim, input_dim);
    layer->dW_logvar = create_matrix(output_dim, input_dim);
    layer->db_mean = (double*)calloc(output_dim, sizeof(double));
    layer->db_logvar = (double*)calloc(output_dim, sizeof(double));
    if (!layer->dW_mean || !layer->dW_logvar || !layer->db_mean || !layer->db_logvar) {
        handle_error("Failed to allocate gradient accumulators.");
    }
    
    // Initialize cached_input pointer to NULL.
    layer->cached_input = NULL;
    
    // Initialize the Prior and Posterior pointers to NULL.
    layer->prior = NULL;
    layer->posterior = NULL;
    
    return layer;
}


// Free the resources allocated for the Bayesian linear layer.
void free_bayesian_linear(BayesianLinear *layer) {
    if (layer) {
        free_matrix(layer->W_mean);
        free_matrix(layer->W_logvar);
        free(layer->b_mean);
        free(layer->b_logvar);
        // Note: The Prior and Posterior objects are managed externally.
        free(layer);
    }
}

Matrix* bayesian_linear_backward(BayesianLinear *layer, const Matrix *grad_output, const Config *cfg) {
    int batch_size = layer->cached_input->rows;
    int in_dim = layer->input_dim;
    int out_dim = layer->output_dim;

    // Clear old gradients
    zero_matrix(layer->dW_mean);
    zero_matrix(layer->dW_logvar);
    zero_array(layer->db_mean, out_dim);
    zero_array(layer->db_logvar, out_dim);

    // Compute gradients from the data loss (MSE component)
    for (int i = 0; i < out_dim; i++) {
        for (int j = 0; j < in_dim; j++) {
            double grad_sum = 0.0;
            for (int b = 0; b < batch_size; b++) {
                grad_sum += grad_output->data[b * out_dim + i] *
                            layer->cached_input->data[b * in_dim + j];
            }
            layer->dW_mean->data[i * in_dim + j] = grad_sum;
        }
        double grad_bias = 0.0;
        for (int b = 0; b < batch_size; b++) {
            grad_bias += grad_output->data[b * out_dim + i];
        }
        layer->db_mean[i] = grad_bias;
    }

    // --- Incorporate the KL divergence gradient ---
    double kl_weight = cfg->kl_weight;  // hardcoded or from cfg
    printf("Inside bayesian_linear_backward: kl_weight = %f\n", kl_weight);
    for (int i = 0; i < out_dim; i++) {
        for (int j = 0; j < in_dim; j++) {
            double old_grad = layer->dW_mean->data[i * in_dim + j];
            double kl_contrib = kl_weight * layer->W_mean->data[i * in_dim + j];
            layer->dW_mean->data[i * in_dim + j] += kl_contrib;
            // Print a few sample gradients for debugging:
            if (i == 0 && j < 5) {
                printf("Grad[%d,%d]: data loss = %f, KL contrib = %f, new grad = %f\n",
                       i, j, old_grad, kl_contrib, layer->dW_mean->data[i * in_dim + j]);
            }
        }
    }
    // Optionally incorporate KL for biases:
    for (int i = 0; i < out_dim; i++) {
        double old_bias_grad = layer->db_mean[i];
        double kl_bias = kl_weight * layer->b_mean[i];
        layer->db_mean[i] += kl_bias;
        if (i < 5) {
            printf("Bias Grad[%d]: data loss = %f, KL contrib = %f, new grad = %f\n",
                   i, old_bias_grad, kl_bias, layer->db_mean[i]);
        }
    }
    fflush(stdout);
    // --------------------------------------------------

    // Compute gradient with respect to inputs.
    Matrix *grad_input = matrix_multiply(grad_output, layer->W_mean);
    return grad_input;
}




// Forward pass for the Bayesian linear layer.
// If 'stochastic' is nonzero, sample weights and biases using the reparameterization trick.
// When a Posterior object is provided, use its sample() function; otherwise, use sample_gaussian().
Matrix* bayesian_linear_forward(BayesianLinear *layer, const Matrix *input, int stochastic) {
    if (input->cols != layer->input_dim) {
        printf("input to cols: %d", input->cols);
        printf("layer to input_dim: %d", layer->input_dim);
        handle_error("Input dimension mismatch in bayesian_linear_forward.");
    }
    // At the start of bayesian_linear_forward():
    if (layer->cached_input) {
        free_matrix(layer->cached_input);
    }
    layer->cached_input = copy_matrix(input); // or implement a copy function
                
    
    int input_samples = input->rows;
    int in_dim = layer->input_dim;
    int out_dim = layer->output_dim;
    
    // Allocate a matrix for the effective weights.
    Matrix *W_effective = create_matrix(out_dim, in_dim);
    
    // Allocate an array for the effective biases.
    double *b_effective = (double*)malloc(sizeof(double) * out_dim);
    if (!b_effective) {
        handle_error("Failed to allocate memory for effective biases.");
    }
    
    // Compute effective weights and biases.
    for (int i = 0; i < out_dim; i++) {
        // Process bias.
        if (stochastic) {
            if (layer->posterior != NULL) {
                b_effective[i] = layer->posterior->sample(layer->posterior, layer->b_mean[i], layer->b_logvar[i]);
            } else {
                b_effective[i] = sample_gaussian(layer->b_mean[i], layer->b_logvar[i]);
            }
        } else {
            b_effective[i] = layer->b_mean[i];
        }
        // Process weights.
        for (int j = 0; j < in_dim; j++) {
            int idx = i * in_dim + j;
            if (stochastic) {
                if (layer->posterior != NULL) {
                    W_effective->data[idx] = layer->posterior->sample(layer->posterior, layer->W_mean->data[idx], layer->W_logvar->data[idx]);
                } else {
                    W_effective->data[idx] = sample_gaussian(layer->W_mean->data[idx], layer->W_logvar->data[idx]);
                }
            } else {
                W_effective->data[idx] = layer->W_mean->data[idx];
            }
        }
    }
    
    // Compute output = input * (W_effective)^T + bias.
    Matrix *W_transposed = matrix_transpose(W_effective);
    Matrix *output = matrix_multiply(input, W_transposed);
    
    free_matrix(W_transposed);
    free_matrix(W_effective);
    
    // Add bias to each row of the output.
    for (int i = 0; i < input_samples; i++) {
        for (int j = 0; j < out_dim; j++) {
            output->data[i * out_dim + j] += b_effective[j];
        }
    }
    free(b_effective);
    return output;
}

// Compute the total KL divergence for the layer's weights and biases using the Prior interface.
// For each weight and bias, if a Prior is set, call its compute_kl() function;
// otherwise, fall back to the Gaussian KL divergence with a default variance of 1.0.
double bayesian_linear_kl(BayesianLinear *layer, double default_variance) {
    double kl_total = 0.0;
    int total_weights = layer->output_dim * layer->input_dim;
    
    if (layer->prior == NULL) {
        // Fallback: use default Gaussian prior with specified variance.
        for (int i = 0; i < total_weights; i++) {
            kl_total += kl_divergence_single(layer->W_mean->data[i],
                                             layer->W_logvar->data[i],
                                             default_variance);
        }
        for (int i = 0; i < layer->output_dim; i++) {
            kl_total += kl_divergence_single(layer->b_mean[i],
                                             layer->b_logvar[i],
                                             default_variance);
        }
    } else {
        // Use the Prior's compute_kl function pointer.
        for (int i = 0; i < total_weights; i++) {
            kl_total += layer->prior->compute_kl(layer->prior,
                                                 layer->W_mean->data[i],
                                                 layer->W_logvar->data[i]);
        }
        for (int i = 0; i < layer->output_dim; i++) {
            kl_total += layer->prior->compute_kl(layer->prior,
                                                 layer->b_mean[i],
                                                 layer->b_logvar[i]);
        }
    }
    return kl_total;
}
