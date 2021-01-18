// ********************************************************************************
// First Serial Implementation of Error Back - Propagation Neural Network Algorithm
// ********************************************************************************

// Include Libraries
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Definitions - Macros


// Declare Arrays
double WL1[100][13];    // First Neural Layer Weights
double WL2[10][101];    // Second Neural Layer Weights
double DL1[100];        // First Neural Hidden Layer
double DL2[10];         // Second Neural Hidden Layer
double OL1[100];        // First Neural Layer Output
double OL2[10];         // Second Neural Layer Output

// ***********************************
// Helper Functions
// ***********************************

double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}
double dSigmoid(double x)
{
    return x * (1 - x);
}
double init_weight(void)
{
    return ((double)rand())/((double)RAND_MAX);
}

// Function to Activate Neural Network
void activateNN(void)                   // TODO: Possibly Add Parameter Option
{

}

// Driver Function
int main(void)
{

    return 0;
}