// ********************************************************************************
// First Serial Implementation of Error Back - Propagation Neural Network Algorithm
// ********************************************************************************

// Include Libraries
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Definitions - Macros
#define HiddenN 100;
#define OutN 10;

// Declare Arrays
double WL1[100][13];    // Hidden Layer Weights
double WL2[10][101];    // Output Layer Weights
double DL1[100];        // Hidden Layer Values
double DL2[10];         // Output Layer Values
double OL1[100];        // Hidden Layer Output
double OL2[10];         // Output Layer Output
double in_vector[12];   // Input Vector
double out_vector[10];  // Output Vector

// ***********************************
// Helper Functions
// ***********************************

// Function to Calculate Using Sigmoid Calculation
double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

// Function to Calculate Using Derivative of Sigmoid Calculation
double dSigmoid(double x)
{
    return x * (1 - x);
}

// Helper Function to Generate Random Weight
double initWeight(void)
{
    return ((double)rand())/((double)RAND_MAX);
}

// Function to Initialize All Weights Using init_weight
void initializeWeights(void)
{
    for (int i = 0; i < 100; i++)
    {
        WL1[i][12] = 1;             // Add Bias

        for (int j = 0; j < 12; j++)
            WL1[i][j] = initWeight();
    }

    for (int i = 0; i < 10; i++)
    {
        WL2[i][100] = 1;            // Add Bias

        for (int j = 0; j < 100; j++)
            WL2[i][j] = initWeight();
    }
}

// Function to Activate Neural Network
void activateNN(void)                   // TODO: Possibly Add Parameter Option
{
    // Forward Pass for Hidden Layer

    for (int i = 0; i < 100; i++)   // For All Neurons in Hidden Layer
    {
        DL1[i] = WL1[i][12];            // Get Bias
        for (int j = 0; j < 12; j++)    // From All Inputs
        {
            DL1[i] += (WL1[i][j] * in_vector[j]);
        }

        OL1[i] = sigmoid(DL1[i]);       // Calculate Output from Sigmoid
    }

    // Forward Pass for Output Layer

    for (int i = 0; i < 10; i++)    // For All Neurons in Output Layer
    {
        DL2[i] = WL2[i][100];           // Get Bias
        for (int j = 0; j < 100; j++)   // From All Neurons in Hidden Layer
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

    for (int i = 0; i < 10; i++)
    {
        temp_error = out_vector[i] - OL2[i];
        total_error += 0.5 * (temp_error * temp_error);
    }

    return total_error;
}

// Function to Train Neural Network
void trainNN(void)
{

}

// Driver Function
int main(void)
{
    srand(time(0));  // Create Seed for rand()

    // Initialize Weights
    initializeWeights();

    return 0;
}