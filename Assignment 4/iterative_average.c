#include <stdio.h>
#include <omp.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

long N = 25165824l, NI = 64;
double *A, *A_shadow;

long ceilDiv(int d, int rank)
{
  long m = N / (long)d;
  if (rank < N % d)
  {
    return (m + 1l);
  }
  else
    return m;
}

long min(long a, long b)
{
  return a < b ? a : b;
}

long getStart(int d, int rank)
{
  long batch = N / ((long)d);
  long start = batch * ((long)rank);
  long extra = N % ((long)d);
  start += min((long)rank, extra);
  return start + 1;
}


int main(int argc, char *argv[])
{
    // Input N(size) and NI(no of iterations) as input
    if(argc>1) {    N = atol(argv[1]);  }
    if(argc>2) {    NI = atol(argv[2]); }

    // printf("N = %ld, NI = %ld\n", N, NI);

    // Declaring the array A using malloc
    A = (double*)malloc(sizeof(double) * (N + 2));
    A_shadow = (double*)malloc(sizeof(double) * (N + 2));

    // Initialize the array
    memset(A, 0, sizeof(double) * (N + 2));
    A[N+1] = N+1;
    memset(A_shadow, 0, sizeof(double) * (N + 2));
    A_shadow[N+1] = N+1;

    int rank, np;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    for(long i=0; i<NI; ++i) 
    {

        long BATCH_SIZE = ceilDiv(np, rank);
        long START = getStart(np, rank);
        long END = START + BATCH_SIZE - 1;

        // printf("My rank = %d, | START = %ld | END = %ld\n", rank, START, END);

        double st = 0.0, en = 0.0;

        if(rank > 0) {
            double x = A[START];
            MPI_Status stats;
            MPI_Send(&x, 1, MPI_DOUBLE, rank-1, i, MPI_COMM_WORLD);
            MPI_Recv(&st, 1, MPI_DOUBLE, rank-1, i, MPI_COMM_WORLD, &stats);
        }

        if(rank < np-1) {
            double x = A[END];
            MPI_Status stats;
            MPI_Send(&x, 1, MPI_DOUBLE, rank+1, i, MPI_COMM_WORLD);
            MPI_Recv(&en, 1, MPI_DOUBLE, rank+1, i, MPI_COMM_WORLD, &stats);
        }

        // Add default(none)
        #pragma omp parallel for
        for(long j=START; j<=END; ++j) {
            double lo = A[j-1], hi = A[j+1];
            if((j == START) && (rank > 0)) {
                // NON BLOCKING VERSION COMMENTED:

                // MPI_Request req;
                MPI_Status stats;
                // MPI_Irecv(&lo, 1, MPI_DOUBLE, rank-1, i, MPI_COMM_WORLD, &req);

                // // Remove this wait for a WaitAll
                // MPI_Wait(&req, &stats);
                // MPI_Recv(&lo, 1, MPI_DOUBLE, rank-1, i, MPI_COMM_WORLD, &stats);
                lo = st;
            }

            if((j == END) && (rank < (np-1))) {
                // NON-BLOCKING VERSION:

                // MPI_Request req;
                MPI_Status stats;
                // MPI_Irecv(&hi, 1, MPI_DOUBLE, rank+1, i, MPI_COMM_WORLD, &req);

                // // Remove this wait for a WaitAll
                // MPI_Wait(&req, &stats);

                // MPI_Recv(&hi, 1, MPI_DOUBLE, rank+1, i, MPI_COMM_WORLD, &stats);
                hi = en;
            }

            A_shadow[j] = (lo+hi) / 2.0;
        }

        // A_shadow[0]=0.0;
        // A_shadow[N+1]=N+1;
        
        // MPI_Irecv needs to be parallelized by waitall and separating it from the loop
        double* temp = A_shadow;
        A_shadow = A;
        A = temp;
        // Using waitall before changing the arrays
        // Used for non-blocking receive

        // for(long i1=0; i1<=N+1; ++i1) printf("%f ", A[i1]);
        // printf("| %d | At end of iteration %ld\n", rank, i);
    }

    // Use OpenMP here with reduction
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for(int i=1; i<=N; ++i) {
        sum+=A[i];
    }

    printf("Rank = %d | SUM = %f\n", rank, sum);

    double total_sum;   // Variable to get total sum
    // Using MPI Reduce here to get the sum from all processors:
    MPI_Reduce(&sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Finalize();


    if(rank == 0) {
        printf("Total Sum: %f\n", total_sum);
    }

    free(A);
    free(A_shadow);
    return 0;
}
