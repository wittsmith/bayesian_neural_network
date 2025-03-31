#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "network.h"
#include "../config/config.h"

// Updates parameters for all layers in the network using a given learning rate.
void network_update_params(Network *net, const Config *cfg);

#endif // OPTIMIZER_H
