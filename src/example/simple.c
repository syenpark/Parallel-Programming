#include <stdio.h>
#include <stdlib.h>

int main()
{
    int *h_A, *h_B, *h_C;
    int i;
    int N = 4096;
    size_t size = N * sizeof(int);

    // Allocate input vectors h_A and h_B in host memory (CPU)
    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (int*)malloc(size);

    // Initialise h_A and h_B here

    // Vector Add
    for (i = 0 ; i < N ; i++)
        h_C[i] = h_A[i] + h_B[i];

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}