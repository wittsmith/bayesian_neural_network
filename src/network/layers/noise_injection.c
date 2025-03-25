#include "noise_injection.h"
#include "../utils/utils.h"        // For handle_error()
#include "../utils/random_utils.h"       // For random_uniform() and random_gaussian()
#include <stdlib.h>

// Create a noise injection module.
NoiseInjection* create_noise_injection(NoiseType type, double mean, double stddev) {
    NoiseInjection *ni = (NoiseInjection*)malloc(sizeof(NoiseInjection));
    if (!ni) {
        handle_error("Failed to allocate NoiseInjection module.");
    }
    ni->type = type;
    ni->mean = mean;
    ni->stddev = stddev;
    return ni;
}

// Free the noise injection module.
void free_noise_injection(NoiseInjection *ni) {
    if (ni) {
        free(ni);
    }
}

// Forward pass for noise injection.
// If training is nonzero, adds noise to each element of the input matrix based on the specified noise type.
// For Gaussian noise: noise is sampled from N(mean, stddev).
// For Uniform noise: noise is sampled from U(mean - stddev, mean + stddev).
// If not training, the input is returned unchanged.
Matrix* noise_injection_forward(NoiseInjection *ni, const Matrix *input, int training) {
    if (!ni || !input) {
        handle_error("Invalid input to noise_injection_forward.");
    }
    
    Matrix *output = create_matrix(input->rows, input->cols);
    int total_elements = input->rows * input->cols;
    
    if (training) {
        for (int i = 0; i < total_elements; i++) {
            double noise = 0.0;
            if (ni->type == NOISE_GAUSSIAN) {
                noise = random_gaussian(ni->mean, ni->stddev);
            } else if (ni->type == NOISE_UNIFORM) {
                // Uniform noise: sample from U(mean - stddev, mean + stddev)
                double u = random_uniform();
                noise = ni->mean - ni->stddev + 2 * ni->stddev * u;
            } else {
                handle_error("Unknown noise type in noise_injection_forward.");
            }
            output->data[i] = input->data[i] + noise;
        }
    } else {
        // In inference mode, return the input unchanged.
        for (int i = 0; i < total_elements; i++) {
            output->data[i] = input->data[i];
        }
    }
    
    return output;
}
