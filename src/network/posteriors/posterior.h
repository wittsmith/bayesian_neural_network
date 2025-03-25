#ifndef POSTERIOR_H
#define POSTERIOR_H

typedef struct Posterior {
    void *data;
    double (*sample)(struct Posterior *posterior, double mu, double logvar);
    double (*compute_kl)(struct Posterior *posterior, double mu, double logvar);
} Posterior;

#endif // POSTERIOR_H
