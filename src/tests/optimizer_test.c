#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "../config/config.h"
#include "../optimizer/optimizer.h"
#include "../optimizer/adam_optimizer.h"
#include <string.h>
// Helper function to check if two doubles are approximately equal
int approx_equal(double a, double b, double tolerance) {
    return fabs(a - b) < tolerance;
}

void test_learning_rate_decay() {
    printf("Testing learning rate decay...\n");
    
    // Test case 1: No decay (lr_decay = 0)
    Config cfg1;
    init_config(&cfg1);
    cfg1.learning_rate = 0.1;
    cfg1.lr_decay = 0.0;
    
    double lr1 = calculate_decayed_lr(&cfg1, 0);
    assert(approx_equal(lr1, 0.1, 1e-6));
    printf("Test 1 passed: No decay case\n");
    
    // Test case 2: With decay
    Config cfg2;
    init_config(&cfg2);
    cfg2.learning_rate = 0.1;
    cfg2.lr_decay = 0.1;
    
    double lr2_epoch0 = calculate_decayed_lr(&cfg2, 0);
    double lr2_epoch1 = calculate_decayed_lr(&cfg2, 1);
    double lr2_epoch2 = calculate_decayed_lr(&cfg2, 2);
    printf("lr2_epoch0: %.6f\n", lr2_epoch0);
    printf("lr2_epoch1: %.6f\n", lr2_epoch1);
    printf("lr2_epoch2: %.6f\n", lr2_epoch2);   
    
    assert(approx_equal(lr2_epoch0, 0.1, 1e-6));
    assert(approx_equal(lr2_epoch1, 0.1 / 1.1, 1e-6));
    assert(approx_equal(lr2_epoch2, 0.1 / 1.2, 1e-6));
    printf("Test 2 passed: With decay case\n");
    
    // Test case 3: Large decay
    Config cfg3;
    init_config(&cfg3);
    cfg3.learning_rate = 0.1;
    cfg3.lr_decay = 1.0;
    
    double lr3_epoch1 = calculate_decayed_lr(&cfg3, 1);
    double lr3_epoch5 = calculate_decayed_lr(&cfg3, 5);
    printf("lr3_epoch1: %.6f\n", lr3_epoch1);
    printf("lr3_epoch5: %.6f\n", lr3_epoch5);
    
    assert(approx_equal(lr3_epoch1, 0.1 / 2.0, 1e-6));
    assert(approx_equal(lr3_epoch5, 0.1 / 6.0, 1e-6));
    printf("Test 3 passed: Large decay case\n");
    
    printf("All learning rate decay tests passed!\n");
}

void test_adam_optimizer() {
    printf("\nTesting Adam optimizer...\n");
    
    // Test case 1: Adam state initialization
    AdamState* state = init_adam_state(5);
    assert(state != NULL);
    assert(state->size == 5);
    assert(state->t == 0);
    assert(state->m != NULL);
    assert(state->v != NULL);
    
    // Verify moment vectors are initialized to zero
    for (int i = 0; i < 5; i++) {
        assert(approx_equal(state->m[i], 0.0, 1e-6));
        assert(approx_equal(state->v[i], 0.0, 1e-6));
    }
    printf("Test 1 passed: Adam state initialization\n");
    
    // Test case 2: Adam update with constant gradient
    Config cfg;
    init_config(&cfg);
    cfg.learning_rate = 0.01;
    cfg.adam_beta1 = 0.9;
    cfg.adam_beta2 = 0.999;
    cfg.adam_epsilon = 1e-8;
    
    // Create test parameters and gradients
    double params[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double grads[5] = {0.1, 0.1, 0.1, 0.1, 0.1};
    
    // Perform multiple updates and verify behavior
    for (int i = 0; i < 3; i++) {
        // Store old parameters
        double old_params[5];
        memcpy(old_params, params, sizeof(params));
        
        // Update parameters
        update_moments_and_params(params, grads, state->m, state->v, 5, &cfg, state->t + 1);
        state->t++;
        
        // Verify parameters have changed
        for (int j = 0; j < 5; j++) {
            assert(params[j] != old_params[j]);
            // Parameters should decrease since gradient is positive
            assert(params[j] < old_params[j]);
        }
    }
    printf("Test 2 passed: Adam parameter updates\n");
    
    // Test case 3: Adam update with alternating gradients
    double alt_grads[5] = {0.1, -0.1, 0.1, -0.1, 0.1};
    double alt_params[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
    
    // Reset state
    state->t = 0;
    memset(state->m, 0, 5 * sizeof(double));
    memset(state->v, 0, 5 * sizeof(double));
    
    // Perform updates with alternating gradients
    for (int i = 0; i < 4; i++) {
        double old_params[5];
        memcpy(old_params, alt_params, sizeof(alt_params));
        
        update_moments_and_params(alt_params, alt_grads, state->m, state->v, 5, &cfg, state->t + 1);
        state->t++;
        
        // Verify parameters change appropriately
        for (int j = 0; j < 5; j++) {
            assert(alt_params[j] != old_params[j]);
            if (alt_grads[j] > 0) {
                assert(alt_params[j] < old_params[j]);
            } else {
                assert(alt_params[j] > old_params[j]);
            }
        }
    }
    printf("Test 3 passed: Adam alternating gradient updates\n");
    
    // Cleanup
    free_adam_state(state);
    printf("All Adam optimizer tests passed!\n");
}

int main() {
    test_learning_rate_decay();
    test_adam_optimizer();
    return 0;
}
