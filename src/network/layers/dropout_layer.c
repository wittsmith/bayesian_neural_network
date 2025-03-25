#include "dropout_layer.h"
#include "../utils/utils.h"         // For handle_error()
#include "../utils/random_utils.h"        // For random_uniform()
#include <stdlib.h>
#include <math.h>

// Helper sigmoid function.
static double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Create a dropout layer.
DropoutLayer* create_dropout_layer(DropoutType type, double dropout_prob, double temperature) {
    DropoutLayer *layer = (DropoutLayer*)malloc(sizeof(DropoutLayer));
    if (!layer) {
        handle_error("Failed to allocate memory for DropoutLayer.");
    }
    layer->type = type;
    layer->dropout_prob = dropout_prob;
    layer->temperature = temperature;
    return layer;
}

// Free the dropout layer.
void free_dropout_layer(DropoutLayer *layer) {
    if (layer) {
        free(layer);
    }
}

// Forward pass for dropout layer.
// Applies dropout element-wise to the input matrix.
// For standard MC dropout: Each element is dropped (set to zero) with probability dropout_prob,
// and the remaining elements are scaled by 1/(1-dropout_prob).
// For concrete dropout: A continuous relaxation is applied.
// The 'training' flag indicates whether to sample a new dropout mask.
Matrix* dropout_forward(DropoutLayer *layer, const Matrix *input, int training) {
    if (!layer || !input) {
        handle_error("Invalid input to dropout_forward.");
    }

    // Create an output matrix with the same dimensions as input.
    Matrix *output = create_matrix(input->rows, input->cols);
    int total_elements = input->rows * input->cols;
    
    // For each element, compute dropout mask and apply it.
    for (int i = 0; i < total_elements; i++) {
        double mask = 1.0; // Default: no dropout
        
        if (layer->type == DROPOUT_MC) {
            // Standard Monte Carlo dropout: sample binary mask.
            // In standard dropout, an element is dropped with probability dropout_prob.
            // We then scale the output by 1/(1-dropout_prob).
            double u = random_uniform();
            if (u < layer->dropout_prob) {
                mask = 0.0;
            } else {
                mask = 1.0 / (1.0 - layer->dropout_prob);
            }
        } else if (layer->type == DROPOUT_CONCRETE) {
            // Concrete dropout:
            // Sample u ~ Uniform(0,1)
            double u = random_uniform();
            // Compute a relaxed mask using the Concrete distribution:
            // s = sigmoid((log(p) - log(1-p) + log(u) - log(1-u)) / temperature)
            // Then use mask = 1 - s to approximate dropout.
            double logit_p = log(layer->dropout_prob) - log(1.0 - layer->dropout_prob);
            double s = sigmoid((logit_p + log(u) - log(1.0 - u)) / layer->temperature);
            mask = 1.0 - s;
            // Optionally, scale by 1/(1-dropout_prob) to maintain the scale.
            mask /= (1.0 - layer->dropout_prob);
        } else {
            handle_error("Unknown dropout type in dropout_forward.");
        }
        
        // Apply the mask to the input.
        output->data[i] = input->data[i] * mask;
    }
    
    return output;
}
