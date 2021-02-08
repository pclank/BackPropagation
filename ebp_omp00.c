// **********************************************************************************
// First Parallel Implementation of Error Back - Propagation Neural Network Algorithm
// **********************************************************************************

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
#define MaxIter 100000

// Declare Arrays
double WL1[HiddenN][InN + 1];   // Hidden Layer Weights
double WL2[OutN][HiddenN + 1];  // Output Layer Weights
double DL1[HiddenN];            // Hidden Layer Values
double DL2[OutN];               // Output Layer Values
double OL1[HiddenN];            // Hidden Layer Output
double OL2[OutN];               // Output Layer Output
double in_vector[InN];          // Training Input Vector
double out_vector[OutN];        // Training Output Vector
double x_test[InN];             // Testing Input Vector
double y_test[OutN];            // Testing Output Vector

const double learn_rate = 0.1f;     // Set Learning Rate
const double max_error = 0.001f; // Set Error to Converge to

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

// Helper Function to Generate Random Input Vector
void generateInput(void)
{
    for (int i = 0; i < InN; i++)
    {
        in_vector[i] = (double)(((double)rand() - RAND_MAX / 2) / (double)RAND_MAX * InMaxValue);
    }
}

// Helper Function to Generate Random Output Vector
void generateOutput(void)
{
    for (int i = 0; i < OutN; i++)
    {
        out_vector[i] = (double)(((double)rand() - RAND_MAX / 2) / (double)RAND_MAX * OutMaxValue);
    }
}

// Helper Function to Print Input and Output Vectors
void printInOut(void)
{
    printf("Printing Input Vector...\n");
    for (int i = 0; i < (InN - 1); i++)
    {
        printf("%f, ", in_vector[i]);
    }
    printf("%f\n\n", in_vector[InN - 1]);

    printf("Printing Output Vector...\n");
    for (int i = 0; i < (OutN - 1); i++)
    {
        printf("%f, ", out_vector[i]);
    }
    printf("%f\n\n", out_vector[OutN - 1]);
}

// Function to Initialize All Weights Using init_weight
void initializeWeights(void)
{
    for (int i = 0; i < HiddenN; i++)
    {
        WL1[i][InN] = 1;             // Add Bias

        for (int j = 0; j < InN; j++)
            WL1[i][j] = initWeight();
    }

    for (int i = 0; i < OutN; i++)
    {
        WL2[i][HiddenN] = 1;            // Add Bias

        for (int j = 0; j < HiddenN; j++)
            WL2[i][j] = initWeight();
    }
}

// Parallel Helper Function to Generate Random Input Vector
void generateInput2(void)
{
    #pragma omp parallel
    {

        int i;

        srand((time(NULL)) ^ omp_get_thread_num());          // Create rand() Seed and Differentiate for Each Thread

        #pragma omp for private(i) schedule(auto)
        for (i = 0; i < InN; i++)
        {
            in_vector[i] = (double) (((double) rand() - RAND_MAX / 2) / (double) RAND_MAX * InMaxValue);
        }
    }
}

// Parallel Helper Function to Generate Random Output Vector
void generateOutput2(void)
{
    #pragma omp parallel
    {
        int i;

        srand((time(NULL)) ^ omp_get_thread_num());          // Create rand() Seed and Differentiate for Each Thread

        #pragma omp for private(i) schedule(auto)
        for (i = 0; i < OutN; i++)
        {
            out_vector[i] = (double) (((double) rand() - RAND_MAX / 2) / (double) RAND_MAX * OutMaxValue);
        }
    }
}

// Parallel Function to Initialize All Weights Using init_weight
void initializeWeights2(void)
{
    int i, j;

    #pragma omp parallel for private(i, j) schedule(auto) collapse(2)
    for (i = 0; i < HiddenN; i++)
    {
        for (j = 0; j <= InN; j++)
            WL1[i][j] = initWeight();
    }

    #pragma omp parallel for private(i, j) schedule(auto) collapse(2)
    for (i = 0; i < OutN; i++)
    {
        for (j = 0; j <= HiddenN; j++)
            WL2[i][j] = initWeight();
    }
}

// Function to Activate Neural Network
void activateNN(void)                   // TODO: Possibly Add Parameter Option
{
    // Forward Pass for Hidden Layer

    for (int i = 0; i < HiddenN; i++)   // For All Neurons in Hidden Layer
    {
        DL1[i] = WL1[i][InN];            // Get Bias
        for (int j = 0; j < InN; j++)    // From All Inputs
        {
            DL1[i] += (WL1[i][j] * in_vector[j]);
        }

        OL1[i] = sigmoid(DL1[i]);       // Calculate Output from Sigmoid
    }

    // Forward Pass for Output Layer

    for (int i = 0; i < OutN; i++)    // For All Neurons in Output Layer
    {
        DL2[i] = WL2[i][HiddenN];           // Get Bias
        for (int j = 0; j < HiddenN; j++)   // From All Neurons in Hidden Layer
        {
            DL2[i] += (WL2[i][j] * OL1[j]);
        }

        OL2[i] = sigmoid(DL2[i]);       // Calculate Output from Sigmoid
    }
}

// Function to Calculate Total Error in Network
double calcError(void)
{
    double total_error = 0;
    double temp_error;

    for (int i = 0; i < OutN; i++)
    {
        temp_error = out_vector[i] - OL2[i];
        total_error += 0.5 * (temp_error * temp_error);
    }

    return total_error;
}

// Function to Train Neural Network
void trainNN(void)
{
    double delta_out[OutN];
    for (int i = 0; i < OutN; i++)
    {
        double error_out = (out_vector[i] - OL2[i]);
        delta_out[i] = (error_out * dSigmoid(OL2[i]));
    }

    double delta_hidden[HiddenN];
    for (int i = 0; i < HiddenN; i++)
    {
        double error_hidden = 0.0f;
        for (int j = 0; j < OutN; j++)
        {
            error_hidden += (delta_out[j] * WL2[j][i]); // TODO: Confirm Correct Indexing
        }

        delta_hidden[i] = (error_hidden * dSigmoid(OL1[i]));
    }

    // Update Output Layer Weights

    for (int i = 0; i < OutN; i++)                  // For All Neurons in Output Layer
    {
        WL2[i][HiddenN] += (delta_out[i] * learn_rate); // Update Bias
        for (int j = 0; j < HiddenN; j++)               // Calculate New Weights from Hidden Layer Neurons
        {
            WL2[i][j] += (OL1[j] * delta_out[i]) * learn_rate;
        }
    }

    // Update Hidden Layer Weights

    for (int i = 0; i < HiddenN; i++)               // For All Neurons in Hidden Layer
    {
        WL1[i][InN] += (delta_hidden[i] * learn_rate);  // Update Bias
        for (int j = 0; j < InN; j++)                   // Calculate New Weights from Input
        {
            WL1[i][j] += (in_vector[j] * delta_hidden[i]) * learn_rate;
        }
    }
}

// Driver Function
int main(void)
{
    srand(time(0));  // Create Seed for rand()

    double total_error = 1;
    int epoch = 1;

    // Generate Random Input
    generateInput2();

    // Generate Random Output
    generateOutput2();

    // Print Generated Vectors
    printInOut();

    // Initialize Weights
    initializeWeights2();

    // Initial Network Activation
    activateNN();

    // Calculate Initial Error
    total_error = calcError();

    printf("Initial Error = %f!\n", total_error);   // Print Initial Activation Error

    // Train Model

    while (total_error > max_error)
    {
        // Update Weights Using Error Back-Propagation
        trainNN();

        // Activate Neurons
        activateNN();

        // Calculate New Error
        total_error = calcError();

//        printf("Epoch %d - Error = %f!\n", epoch, total_error);  // Print Epoch Information

        epoch++;    // Increment Epoch Variable

        if (epoch > MaxIter)
        {
            break;
        }
    }

    printf("Final Error was %f!", total_error);

    return 0;
}