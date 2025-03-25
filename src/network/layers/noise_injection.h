#ifndef NOISE_INJECTION_H
#define NOISE_INJECTION_H

#include "../utils/math_utils.h"  // For the Matrix type

// Enumeration for different types of noise.
typedef enum {
    NOISE_GAUSSIAN,
    NOISE_UNIFORM
} NoiseType;

// Structure representing a noise injection module.
typedef struct {
    NoiseType type;  // The type of noise (currently supporting Gaussian and Uniform)
    double mean;     // Mean of the noise (for Gaussian noise, typically 0)
    double stddev;   // Standard deviation of the noise (for Gaussian noise)
    // For uniform noise, you could interpret stddev as half-range.
} NoiseInjection;

// Create a noise injection module.
NoiseInjection* create_noise_injection(NoiseType type, double mean, double stddev);

// Free the noise injection module.
void free_noise_injection(NoiseInjection *ni);

// Forward pass for noise injection: adds noise element-wise to the input matrix.
// If training is nonzero, noise is added; if not, the input is passed unchanged.
Matrix* noise_injection_forward(NoiseInjection *ni, const Matrix *input, int training);

#endif // NOISE_INJECTION_H
