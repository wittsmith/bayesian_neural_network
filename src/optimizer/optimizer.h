#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "network.h"

// Updates parameters for all layers in the network using a given learning rate.
void network_update_params(Network *net, double lr);

#endif // OPTIMIZER_H
