// *******************************************************************************************************
// Parallel Implementation of Error Back - Propagation Neural Network Algorithm Based on Input Rows - Sets
// *******************************************************************************************************

// Include Libraries
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

// Definitions - Macros
#define HiddenN 100
#define OutN 10
#define InN 12
#define InMaxValue 1
#define OutMaxValue 1
#define MaxIter 10000
#define TrainingSets 4

// Declare Arrays
double WL1[HiddenN][InN + 1];                   // Hidden Layer Weights
double WL2[OutN][HiddenN + 1];                  // Output Layer Weights
double DL1[HiddenN];                            // Hidden Layer Values
double DL2[OutN];                               // Output Layer Values
double OL1[HiddenN];                            // Hidden Layer Output
double OL2[OutN];                               // Output Layer Output
double training_inputs[TrainingSets][InN];      // Training Inputs
double training_outputs[TrainingSets][OutN];    // Training Outputs
int training_order[TrainingSets];               // Order of Processing per Epoch

const double learn_rate = 0.1f;     // Set Learning Rate

// *******************************************************************
#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline", "unsafe-math-optimizations")
#pragma GCC option("arch=native","tune=native","no-zero-upper")
//************************************************************

// ***********************************
// Helper Functions
// ***********************************

// Helper Function to Calculate Using Sigmoid Calculation
double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

// Helper Function to Calculate Using Derivative of Sigmoid Calculation
double dSigmoid(double x)
{
    return x * (1 - x);
}

// Helper Function to Generate Random Weight
double initWeight(void)
{
    return ((double)rand())/((double)RAND_MAX);
}

// Parallel Helper Function to Generate Random Input Vector with Input Set Parameter
void generateInput2(void)
{
    #pragma omp parallel
    {
        int i, j;

        srand((time(NULL)) ^ omp_get_thread_num());          // Create rand() Seed and Differentiate for Each Thread

        #pragma omp for private(i, j) schedule(auto) collapse(2)
        for (i = 0; i < TrainingSets; i++)
        {
            for (j = 0; j < InN; j++)
                training_inputs[i][j] = (double) (((double) rand() - RAND_MAX / 2) / (double) RAND_MAX * InMaxValue);
        }
    }
}

// Parallel Helper Function to Generate Random Output Vector with Input Set Parameter
void generateOutput2(void)
{
    #pragma omp parallel
    {
        int i, j;

        srand((time(NULL)) ^ omp_get_thread_num());          // Create rand() Seed and Differentiate for Each Thread

        #pragma omp for private(i, j) schedule(auto) collapse(2)
        for (i = 0; i < TrainingSets; i++)
        {
            for (j = 0; j < OutN; j++)
                training_outputs[i][j] = (double) (((double) rand() - RAND_MAX / 2) / (double) RAND_MAX * OutMaxValue);
        }
    }
}

// Helper Function to Print Input and Output Vectors with Input Set Parameter
void printInOut2(void)
{
    printf("Printing Input Vector...\n");
    for (int j = 0; j < TrainingSets; j++)
    {
        printf("Set %d: ", j);
        for (int i = 0; i < (InN - 1); i++)
        {
            printf("%f, ", training_inputs[j][i]);
        }
        printf("%f\n\n", training_inputs[j][InN - 1]);
    }

    printf("Printing Output Vector...\n");
    for (int j = 0; j < TrainingSets; j++)
    {
        printf("Set %d: ", j);
        for (int i = 0; i < (OutN - 1); i++)
        {
            printf("%f, ", training_outputs[j][i]);
        }
        printf("%f\n\n", training_outputs[j][OutN - 1]);
    }
}

void shuffle(int *array, size_t n)
{
    if (n > 1)
    {
        size_t i;
        for (i = 0; i < n - 1; i++)
        {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

// Parallel Function to Initialize All Weights Using init_weight
void initializeWeights(void)
{
    int i, j;

    #pragma omp parallel for private(i, j) schedule(auto) collapse(2)
    for (i = 0; i < HiddenN; i++)
    {
//        WL1[i][InN] = 1;             // Add Bias

        for (j = 0; j <= InN; j++)
            WL1[i][j] = initWeight();
    }

    #pragma omp parallel for private(i, j) schedule(auto) collapse(2)
    for (i = 0; i < OutN; i++)
    {
//        WL2[i][HiddenN] = 1;            // Add Bias

        for (j = 0; j <= HiddenN; j++)
            WL2[i][j] = initWeight();
    }
}

// Function to Activate Neural Network with Input Set Parameter
void activateNN2(int set)
{
    int i, j;

    // Forward Pass for Hidden Layer

//    #pragma omp parallel for private(i, j) schedule(auto) collapse(2)
    for (i = 0; i < HiddenN; i++)   // For All Neurons in Hidden Layer
    {
        for (j = 0; j < InN; j++)    // From All Inputs
        {
            if (j == 0)
            {
                DL1[i] = WL1[i][InN];            // Get Bias
            }

            DL1[i] += (WL1[i][j] * training_inputs[set][j]);

            if (j == (InN - 1))
            {
                OL1[i] = sigmoid(DL1[i]);       // Calculate Output from Sigmoid
            }
        }
    }

    // Forward Pass for Output Layer

//    #pragma omp parallel for private(i, j) schedule(auto) collapse(2)
    for (i = 0; i < OutN; i++)    // For All Neurons in Output Layer
    {
        for (j = 0; j < HiddenN; j++)   // From All Neurons in Hidden Layer
        {
            if (j == 0)
            {
                DL2[i] = WL2[i][HiddenN];           // Get Bias
            }

            DL2[i] += (WL2[i][j] * OL1[j]);

            if (j == (HiddenN - 1))
            {
                OL2[i] = sigmoid(DL2[i]);       // Calculate Output from Sigmoid
            }
        }
    }
}

// Function to Calculate Total Error in Network with Input Set Parameter
double calcError2(int set)
{
    double total_error = 0;
    double temp_error;

    for (int i = 0; i < OutN; i++)
    {
        temp_error = training_outputs[set][i] - OL2[i];
        total_error += 0.5 * (temp_error * temp_error);
    }

    return total_error;
}

// Parallel Function to Train Neural Network with Input Set Parameter
void trainNN2(int set)
{
    int i, j;
    double delta_out[OutN];
    double delta_hidden[HiddenN];

    #pragma omp parallel
    {

        #pragma omp for private(i)
        for (i = 0; i < OutN; i++)
        {
            double error_out = (training_outputs[set][i] - OL2[i]);
            delta_out[i] = (error_out * dSigmoid(OL2[i]));
        }

        #pragma omp barrier

        #pragma omp for private(i, j)
        for (i = 0; i < HiddenN; i++)
        {
            double error_hidden = 0.0f;
            for (j = 0; j < OutN; j++) {
                error_hidden += (delta_out[j] * WL2[j][i]);
            }

            delta_hidden[i] = (error_hidden * dSigmoid(OL1[i]));
        }
    }

    #pragma omp parallel
    {
        // Update Output Layer Weights

        #pragma omp for private(i, j) nowait
        for (i = 0; i < OutN; i++)                  // For All Neurons in Output Layer
        {
            WL2[i][HiddenN] += (delta_out[i] * learn_rate); // Update Bias
            for (j = 0; j < HiddenN; j++)               // Calculate New Weights from Hidden Layer Neurons
            {
                WL2[i][j] += (OL1[j] * delta_out[i]) * learn_rate;
            }
        }

        // Update Hidden Layer Weights

        #pragma omp for private(i, j)
        for (i = 0; i < HiddenN; i++)               // For All Neurons in Hidden Layer
        {
            WL1[i][InN] += (delta_hidden[i] * learn_rate);  // Update Bias
            for (j = 0; j < InN; j++)                   // Calculate New Weights from Input
            {
                WL1[i][j] += (training_inputs[set][j] * delta_hidden[i]) * learn_rate;
            }
        }
    }
}

// Driver Function
int main(void)
{
    srand(time(0));  // Create Seed for rand()

    double total_error;

    // Generate Random Input
    generateInput2();

    // Generate Random Output
    generateOutput2();

    // Print Generated Vectors
    printInOut2();

    // Initialize Weights
    initializeWeights();

    for (int epoch = 1; epoch < MaxIter; epoch++)
    {
        total_error = 0;                            // Reset Total Error for Epoch

        shuffle(training_order, TrainingSets);      // Shuffle Input Sets
        for (int j = 0; j < TrainingSets; j++)
        {
            int set = training_order[j];

            activateNN2(set);

            // Update Weights Using Error Back-Propagation
            trainNN2(set);

            // Calculate New Error
            total_error += calcError2(set);

//            printf("Epoch %d - Error = %f!\n", epoch, total_error);  // Print Epoch Information
        }

        if (epoch == 1)
        {
            printf("Initial Error = %f!\n", total_error);   // Print Initial Activation Error
        }
    }

    printf("Final Error was %f!", total_error);

    return 0;
}