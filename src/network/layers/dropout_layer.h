#ifndef DROPOUT_LAYER_H
#define DROPOUT_LAYER_H

#include "../utils/math_utils.h"  // For the Matrix type

// Enumeration to distinguish between dropout variants.
typedef enum {
    DROPOUT_MC,        // Standard Monte Carlo dropout
    DROPOUT_CONCRETE   // Concrete dropout with learnable probability
} DropoutType;

// Structure representing a dropout layer.
typedef struct {
    DropoutType type;       // Type of dropout (MC or Concrete)
    double dropout_prob;    // Dropout probability (for MC dropout, fixed; for concrete, learned)
    double temperature;     // Temperature parameter for concrete dropout (ignored for MC dropout)
} DropoutLayer;

// Create a dropout layer with the specified type, dropout probability, and temperature.
// For MC dropout, temperature can be set to any value (e.g., 0.0).
DropoutLayer* create_dropout_layer(DropoutType type, double dropout_prob, double temperature);

// Free the memory allocated for a dropout layer.
void free_dropout_layer(DropoutLayer *layer);

// Forward pass for the dropout layer.
// 'input' is a pointer to a Matrix containing activations.
// 'training' flag can be used to decide whether to sample a new mask.
// Note: For MC dropout in a BNN, dropout is kept active at inference, so typically
// the same function is used regardless of training/inference mode.
Matrix* dropout_forward(DropoutLayer *layer, const Matrix *input, int training);

#endif // DROPOUT_LAYER_H
