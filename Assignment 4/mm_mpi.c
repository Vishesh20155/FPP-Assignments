#include <stdio.h>
#include <omp.h>
#include <mpi.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <assert.h>

int N = 1024;

int main(int argc, char *argv[])
{
    if(argc > 1)    N = atoi(argv[1]);

    int A[N][N], B[N][N], C[N][N];
    int rank, np;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Initialization of A & B in Master thread:
    if(rank == 0)
    {
        for(int i=0; i<N; ++i)
        {
            memset(A, 1, sizeof(int)*N);
            memset(B, 1, sizeof(int)*N);
            memset(C, 0, sizeof(int)*N);
        }
    }

    printf("Inside Proc %d\n", rank);


    // Validating the output, checking against values as in Assignment 1:
    if(rank == 0)
    {
        for(int i=0; i<N; i++) for(int j=0; j<N; j++) assert(C[i][j] == N);
        printf("Test Success. \n");
    }
    MPI_Finalize();

    return 0;
}
