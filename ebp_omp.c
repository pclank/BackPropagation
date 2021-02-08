// ***********************************************************************************************
// First Parallel Implementation of Error Back - Propagation Neural Network Algorithm Using OpenMP
// ***********************************************************************************************

// Include Libraries
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

// Definitions - Macros
#define HiddenN 2
#define OutN 1
#define InN 2
#define InMaxValue 1
#define OutMaxValue 1
#define MaxIter 100000
#define TrainingSets 4

// Declare Arrays
double WL1[HiddenN][InN + 1];   // Hidden Layer Weights
double WL2[OutN][HiddenN + 1];  // Output Layer Weights
double DL1[HiddenN];            // Hidden Layer Values
double DL2[OutN];               // Output Layer Values
double OL1[HiddenN];            // Hidden Layer Output
double OL2[OutN];               // Output Layer Output
double in_vector[InN];          // Training Input Vector
double out_vector[OutN];        // Training Output Vector

double training_inputs[TrainingSets][InN] = { {0.0f,0.0f},{1.0f,0.0f},{0.0f,1.0f},{1.0f,1.0f} };
double training_outputs[TrainingSets][OutN] = { {0.0f},{1.0f},{1.0f},{0.0f} };

const double learn_rate = 0.1f;     // Set Learning Rate
const double max_error = 0.001f;    // Set Error to Converge to

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

// Parallel Helper Function to Generate Random Input Vector
void generateInput(void)
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
void generateOutput(void)
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

// Function to Activate Neural Network with Input Set Parameter
void activateNN2(int set)
{
    // Forward Pass for Hidden Layer

    for (int i = 0; i < HiddenN; i++)   // For All Neurons in Hidden Layer
    {
        DL1[i] = WL1[i][InN];            // Get Bias
        for (int j = 0; j < InN; j++)    // From All Inputs
        {
            DL1[i] += (WL1[i][j] * training_inputs[set][j]);
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
    for (i = 0; i < OutN; i++)
    {
        double error_out = (training_outputs[set][i] - OL2[i]);
        delta_out[i] = (error_out * dSigmoid(OL2[i]));
    }

    double delta_hidden[HiddenN];
    for (i = 0; i < HiddenN; i++)
    {
        double error_hidden = 0.0f;
        for (j = 0; j < OutN; j++)
        {
            error_hidden += (delta_out[j] * WL2[j][i]);
        }

        delta_hidden[i] = (error_hidden * dSigmoid(OL1[i]));
    }

    // Update Output Layer Weights

    for (i = 0; i < OutN; i++)                  // For All Neurons in Output Layer
    {
        WL2[i][HiddenN] += (delta_out[i] * learn_rate); // Update Bias
        for (j = 0; j < HiddenN; j++)               // Calculate New Weights from Hidden Layer Neurons
        {
            WL2[i][j] += (OL1[j] * delta_out[i]) * learn_rate;
        }
    }

    // Update Hidden Layer Weights

    for (i = 0; i < HiddenN; i++)               // For All Neurons in Hidden Layer
    {
        WL1[i][InN] += (delta_hidden[i] * learn_rate);  // Update Bias
        for (j = 0; j < InN; j++)                   // Calculate New Weights from Input
        {
            WL1[i][j] += (training_inputs[set][j] * delta_hidden[i]) * learn_rate;
        }
    }
}

// Driver Function
int main(void)
{
    srand(time(0));  // Create Seed for rand()

    double total_error = 1;

    // Initialize Weights
    initializeWeights();

    int training_order[] = {0, 1, 2, 3};

    for (int epoch = 1; epoch < MaxIter; epoch++)
    {
        shuffle(training_order, TrainingSets);
        for (int j = 0; j < TrainingSets; j++)
        {
            int set = training_order[j];

            activateNN2(set);

            // Update Weights Using Error Back-Propagation
            trainNN2(set);

            // Calculate New Error
            total_error = calcError2(set);

            printf("Epoch %d - Error = %f!\n", epoch, total_error);  // Print Epoch Information
        }
    }

    return 0;
}