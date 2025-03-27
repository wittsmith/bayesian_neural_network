#include "random_utils.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>

void init_random(unsigned int seed) {
    srand(seed);
}

double random_uniform() {
    return ((double)rand() + 1.0) / ((double)RAND_MAX + 1.0);
}


double random_gaussian(double mean, double stddev) {
    // Use Box-Muller transform to generate a standard normal random value.
    double u1 = random_uniform();
    double u2 = random_uniform();
    double z0 = sqrt(-2.0 * log(u1)) * cos(2 * M_PI * u2);
    return z0 * stddev + mean;
}

int random_bernoulli(double p) {
    return (random_uniform() < p) ? 1 : 0;
}
