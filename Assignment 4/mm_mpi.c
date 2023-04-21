#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <assert.h>

int N = 1024;
int TAG1 = 17, TAG2 = 45, TAG3 = 18;

// Min implementation for int parameters
long min(int a, int b)
{
  return a < b ? a : b;
}

long get_usecs () {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec*1000000+t.tv_usec;
}

int getBatch(int d, int rank)
{
    int m = N / d;
    if (rank < N % d)
    {
        return (m + 1l);
    }
    else
        return m;
}

// Function to compute the starting index for inner loop for current rank
int getStart(int d, int rank)
{
  int batch = N / (d);
  int start = batch * (rank);
  int extra = N % (d);
  start += min(rank, extra);
  return start;
}

int main(int argc, char *argv[])
{
    if (argc > 1)
        N = atoi(argv[1]);

    int rank, np;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int batchSize = getBatch(np, rank);
    int startIndex = getStart(np, rank);
    // printf("Batch Size %d and Starting row %d in rank %d\n", batchSize, startIndex, rank);

    // Efficient Allocation of A & C in each proc:
    int A[rank == 0 ? N : batchSize][N], B[N][N], C[rank == 0 ? N : batchSize][N];
    int start_timer, end_timer;
    // Initialization of A & B in Master thread:
    if (rank == 0)
    {
        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                A[i][j] = 1;
                B[i][j] = 1;
                C[i][j] = 0;
            }
        }

        start_timer = get_usecs();
        // Master sends the Matrices A, B & C
        for (int r = 1; r < np; ++r)
        {
            MPI_Send(A + getStart(np, r), N*getBatch(np, r), MPI_INT, r, TAG1, MPI_COMM_WORLD);
            MPI_Send(B, N * N, MPI_INT, r, TAG2, MPI_COMM_WORLD);
        }
    }
    else    // Processors other than Master receive A&B
    {
        // Do an non-blocking receive here
        MPI_Status stats1, stats2;
        MPI_Recv(A, batchSize * N, MPI_INT, 0, TAG1, MPI_COMM_WORLD, &stats1);
        MPI_Recv(B, N * N, MPI_INT, 0, TAG2, MPI_COMM_WORLD, &stats2);

        // Initialize C while receive is Non-Blocking
        for(int i=0; i<batchSize; ++i) 
        {
            for(int j=0; j<N; ++j)
            {
                C[i][j] = 0;
            }
        }

        // Use wait before starting computation
    }

    
    // Computing MatMul here:
    for(int i=0; i<batchSize; ++i) 
    {
        for(int j=0; j<N; ++j) 
        {
            for(int k=0; k<N; ++k) 
            {
                C[i][j] += (A[i][k] * B[k][j]);
            }
        }
    }

/*  // For printing inside a proc

    printf("Inside Proc %d Matrix C:\n", rank);
    for (int i = 0; i < batchSize; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            printf("%d ", C[i][j]);
        }
        printf("\t:%d\n", rank);
    }
*/

    // Send the result to Master from other threads:
    if(rank > 0) 
    {
        // Send NON BLOCKING here
        MPI_Send(C, batchSize*N, MPI_INT, 0, TAG3, MPI_COMM_WORLD);
    }
    else    // Receive in Master
    {
        // Receive NON-BLOCKING here
        for(int r=1; r<np; ++r)
        {
            MPI_Status stats;
            MPI_Recv(C + getStart(np, r), getBatch(np, r)*N, MPI_INT, r, TAG3, MPI_COMM_WORLD, &stats);
        }

        // Put a WaitAll here

        end_timer = get_usecs();
        double dur = ((double)(end_timer-start_timer))/1000000;
        // Validating the output, checking against values as in Assignment 1:
        for(int i=0; i<N; i++) for(int j=0; j<N; j++) assert(C[i][j] == N);
        printf("Test Success. \n");
        printf("Time = %.3f\n", dur);
    }
    
    MPI_Finalize();

    return 0;
}
