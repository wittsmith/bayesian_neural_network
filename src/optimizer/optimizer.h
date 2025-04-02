#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "network.h"
#include "../config/config.h"

// Calculate the decayed learning rate based on the current epoch
double calculate_decayed_lr(const Config *cfg, int current_epoch);

// Updates parameters for all layers in the network using a given learning rate.
void network_update_params(Network *net, const Config *cfg, int current_epoch);

#endif // OPTIMIZER_H
