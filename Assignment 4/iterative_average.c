#include <stdio.h>
#include <omp.h>
#include <mpi.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>

long N = 25165824l, NI = 64;
double *A, *A_shadow;

// Function to find Batch Size for corresponding rank:
/*
    Eg:
    N = 22, np = 4
    will be broken as:
        6 + 6 + 5 + 5
    batch sizes
*/
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

// Min implementation for long parameters
long min(long a, long b)
{
  return a < b ? a : b;
}

// Function to compute the starting index for inner loop for current rank
long getStart(int d, int rank)
{
  long batch = N / ((long)d);
  long start = batch * ((long)rank);
  long extra = N % ((long)d);
  start += min((long)rank, extra);
  return start + 1;
}

long get_usecs () {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec*1000000+t.tv_usec;
}


int main(int argc, char *argv[])
{
    // Input N(size) and NI(no of iterations) as input
    if(argc>1) {    N = atol(argv[1]);  }
    if(argc>2) {    NI = atol(argv[2]); }

    // printf("N = %ld, NI = %ld\n", N, NI);

    int rank, np;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    long BATCH_SIZE = ceilDiv(np, rank);

    long array_size = ( ((rank == 0) || (rank == np-1)) ? BATCH_SIZE + 1 : BATCH_SIZE );
    // printf("Array Size %ld in rank %d\n", array_size, rank);

    // Declaring the array A using malloc
    A = (double*)malloc(sizeof(double) * array_size);
    A_shadow = (double*)malloc(sizeof(double) * array_size);

    // Initialize the array
    memset(A, 0, sizeof(double) * array_size);
    memset(A_shadow, 0, sizeof(double) * array_size);
    if(rank == np-1)
    {
        A[array_size-1] = N+1;
        A_shadow[array_size-1] = N+1;
    }

    long START = ( (rank == 0) ? 1 : 0 );
    long END = ( (rank == np-1) ? array_size-2 : array_size-1);

    // Start timer here:
    long start_timer = get_usecs();
    
    for(long i=0; i<NI; ++i) 
    {
        // printf("My rank = %d, | START = %ld | END = %ld\n", rank, START, END);

        double st = 0.0, en = 0.0;

        if(rank > 0) {
            double x = A[START];
            MPI_Request req;
            MPI_Status stats;
            MPI_Isend(&x, 1, MPI_DOUBLE, rank-1, i, MPI_COMM_WORLD, &req);
            MPI_Wait(&req, &stats);
            // MPI_Send(&x, 1, MPI_DOUBLE, rank-1, i, MPI_COMM_WORLD);
            // MPI_Recv(&st, 1, MPI_DOUBLE, rank-1, i, MPI_COMM_WORLD, &stats);
        }

        if(rank < np-1) {
            double x = A[END];
            MPI_Request req;
            MPI_Status stats;
            MPI_Isend(&x, 1, MPI_DOUBLE, rank+1, i, MPI_COMM_WORLD, &req);
            MPI_Wait(&req, &stats);
            // MPI_Send(&x, 1, MPI_DOUBLE, rank+1, i, MPI_COMM_WORLD);
            // MPI_Recv(&en, 1, MPI_DOUBLE, rank+1, i, MPI_COMM_WORLD, &stats);
        }

        
        MPI_Request startReq, endReq;
        MPI_Status startStatus, endStatus;
        int startCompute = 1, endCompute = 1;

        // Add default(none)
        #pragma omp parallel for
        for(long j=START; j<=END; ++j) 
        {
            int sc=1, ec=1;
            double lo = A[j-1], hi = A[j+1];
            if((j == START) && (rank > 0)) 
            {
                startCompute = 0; sc = 0;
                // NON BLOCKING VERSION:
                MPI_Status stats;
                MPI_Irecv(&st, 1, MPI_DOUBLE, rank-1, i, MPI_COMM_WORLD, &startReq);
            }

            if((j == END) && (rank < (np-1))) 
            {
                endCompute = 0; ec = 0;
                // NON-BLOCKING VERSION:

                MPI_Status stats;
                MPI_Irecv(&en, 1, MPI_DOUBLE, rank+1, i, MPI_COMM_WORLD, &endReq);
            }

            // Case when we have both A[j-1] & A[j+1]:
            if (sc == 1 && ec == 1)
            {
                A_shadow[j] = (lo + hi) / 2.0;
            }
        }

    
        // Handling the case when the chunk's first elements computation remains due to communication:
        if (startCompute == 0) 
        {
            MPI_Wait(&startReq, &startStatus);
            if(START == END) 
            {
                // Case when A[j+1] is also not available due to NON-BLOCKING 
                if(endCompute == 0) 
                {
                    MPI_Wait(&endReq, &endStatus);
                    A_shadow[END] = (st + en) / 2.0;
                    endCompute = 1;
                }
                else
                {
                    A_shadow[END] = (st + A[START+1]) / 2.0;
                }
                startCompute = 1;
            }
            else 
            {
                A_shadow[START] = (st + A[START+1]) / 2.0;
                startCompute = 1;
            }
        }
        
        // Handling the case when the chunk's first elements computation remains due to communication:
        if(endCompute == 0) 
        {
            // printf("EndCompute in iteration %ld in rank %d\n", i, rank);
            MPI_Wait(&endReq, &endStatus);
            A_shadow[END] = (A[END-1] + en) / 2.0;
            endCompute = 1;
        }


        /*
            WaitAll has not been used because in WaitAll, 
            we have to specify array for req and status of size 2 
            but we will not always require sized 2 
            (in case of corner chunks)
        */
    
        
        // MPI_Irecv needs to be parallelized by waitall and separating it from the loop
        double* temp = A_shadow;
        A_shadow = A;
        A = temp;
        // Using waitall before changing the arrays
        // Used for non-blocking receive

    /*      //For PRINTING array A after each iteration

        for(long i1=0; i1<=N+1; ++i1) printf("%f ", A[i1]);
        printf("| %d | At end of iteration %ld\n", rank, i);
    */
    }

    // Use OpenMP here with reduction
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for(long i=START; i<=END; ++i) {
        sum+=A[i];
    }

    // printf("Rank = %d | SUM = %f\n", rank, sum);

    double total_sum;   // Variable to get total sum

    // Using MPI Reduce here to get the sum from all processors:
    MPI_Reduce(&sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    long end_timer = get_usecs();
    double dur = ((double)(end_timer-start_timer))/1000000;

    free(A);
    free(A_shadow);

    MPI_Finalize();


    if(rank == 0) {
        printf("Total Sum: %f\n", total_sum);
        printf("Time = %.3f\n",dur);
    }

    return 0;
}
