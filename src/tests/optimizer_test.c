#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "../config/config.h"
#include "../optimizer/optimizer.h"

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

int main() {
    test_learning_rate_decay();
    return 0;
}
