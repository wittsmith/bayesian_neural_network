#ifndef DATA_H
#define DATA_H

// Dataset structure:
// - num_samples: number of examples in the dataset.
// - num_features: number of features per example (assumes the CSV’s last column is the label).
// - X: pointer to a contiguous block of doubles storing the features in row–major order.
// - y: pointer to an array of doubles storing the labels.
typedef struct {
    int num_samples;
    int num_features;
    double *X;  // Dimensions: num_samples x num_features.
    double *y;  // Dimensions: num_samples.
} Dataset;

// Load a CSV file into a Dataset structure.
// 'filename' is the path to the CSV file.
// 'has_header' should be 1 if the CSV file contains a header row, 0 otherwise.
// Assumes that each row contains N columns with the last column being the label.
Dataset* load_csv(const char *filename, int has_header);

// Free memory allocated for a Dataset.
void free_dataset(Dataset *ds);

// Shuffle the dataset in-place using Fisher–Yates algorithm.
void shuffle_dataset(Dataset *ds);

// Extract a mini–batch from the dataset starting at 'start' with 'batch_size' examples.
// Returns a new Dataset object that must be freed by free_dataset().
Dataset* get_minibatch(const Dataset *ds, int start, int batch_size);

#endif // DATA_H
