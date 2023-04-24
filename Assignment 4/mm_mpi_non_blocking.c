#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <sys/time.h>
#include <assert.h>

int N = 1024;
int TAG1 = 0, TAG2 = 2000, TAG3 = 4000;

// Min implementation for int parameters
int min(int a, int b)
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
    if(argc > 1) 
    {
        N = atoi(argv[1]);
    }

    int rank, np;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int batchSize = getBatch(np, rank);
    int startIndex = getStart(np, rank);
    int numRows = ( (rank == 0) ? N : batchSize );

    // printf("Start Index %d | Batch Size %d | Rank %d\n", startIndex, batchSize, rank);

    // Efficient declaration of A in each rank
    int **A = (int **)malloc(numRows * sizeof(int *));
    for(int i=0; i<numRows; ++i)
    {
        A[i] = (int *)malloc(N * sizeof(int));
    }

    int **B = (int **)malloc(N * sizeof(int *));
    for(int i=0; i<N; ++i) 
    {
        B[i] = (int *)malloc(N * sizeof(int));
    }

    // Allocating C efficiently -- only required number of rows are allocated
    int **C = (int **)malloc(numRows * sizeof(int *));
    for(int i=0; i<numRows; ++i)
    {
        C[i] = (int *)malloc(N * sizeof(int));
    }

    MPI_Request *reqsA;
    if(rank > 0)
        reqsA = (MPI_Request*)malloc(numRows * sizeof(MPI_Request));

    MPI_Status *statsA;
    if(rank > 0)
        statsA = (MPI_Status*)malloc(numRows * sizeof(MPI_Status));

    MPI_Request *reqsC;
    if(rank == 0)
        reqsC = (MPI_Request*)malloc((N-batchSize) * sizeof(MPI_Request));

    MPI_Status *statsC;
    if(rank == 0)
        statsC = (MPI_Status*)malloc((N-batchSize) * sizeof(MPI_Status));

    long start_time, end_time;


    if(rank == 0)
    {
        // Initializing A & B in Master
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i][j] = 1;
                B[i][j] = 1;
            }
        }

        start_time = get_usecs();
        for(int rnk=1; rnk<np; ++rnk)
        {
            int rankStart = getStart(np, rnk);
            int rankBatch = getBatch(np, rnk);
            // for(int i=0; i<rankBatch; ++i)
            //     MPI_Send(A[i+rankStart], N, MPI_INT, rnk, TAG1 + i, MPI_COMM_WORLD);

            // Non-Blocking send of A
            MPI_Request reqs[rankBatch];
            for(int i=0; i<rankBatch; ++i) 
            {
                MPI_Isend(A[i+rankStart], N, MPI_INT, rnk, TAG1+i, MPI_COMM_WORLD, &reqs[i]);
            }
        }
    }
    else 
    {
        // MPI_Status stat[batchSize];
        // for(int i=0; i<batchSize; ++i)
        //     MPI_Recv(A[i], N, MPI_INT, 0, TAG1 + i, MPI_COMM_WORLD, &stat[i]);
        for(int i=0; i<batchSize; ++i)
            MPI_Irecv(A[i], N, MPI_INT, 0, TAG1+i, MPI_COMM_WORLD, &reqsA[i]);
    }

    // Get value of B using Broadcasr
    for(int i=0; i<N; ++i)
        MPI_Bcast(B[i], N, MPI_INT, 0, MPI_COMM_WORLD);

    // Initializing C in each rank:
    for(int i=0; i<numRows; ++i)
    {
        for(int j=0; j<N; ++j)
        {
            C[i][j] = 0;
        }
    }

    // Computing MatMul in C
    for(int i=0; i<batchSize; ++i)
    {
        // Ensuring A[i] has been received by using wait:
        if(rank > 0)
        {
            MPI_Wait(&reqsA[i], &statsA[i]);
        }
        for(int j=0; j<N; ++j)
        {
            for(int k=0; k<N; ++k)
            {
                C[i][j] += (A[i][k] * B[k][j]);
            }
        }

        if(rank != 0)
        {
            // MPI_Send(C[i], N, MPI_INT, 0, TAG3+i, MPI_COMM_WORLD);

            // NON BLOCKING send of C
            MPI_Request req;
            MPI_Isend(C[i], N, MPI_INT, 0, TAG3+i, MPI_COMM_WORLD, &req);
        }
    }

    if (rank == 0) 
    {
        // Master (root) Receiving the data in a Non-Blocking Call
        for(int rnk=1; rnk<np; ++rnk)
        {
            int rankStart = getStart(np, rnk);
            int rankBatch = getBatch(np, rnk);

            // Non Blocking Receive of C
            for(int i=0; i<rankBatch; ++i)
            {
                MPI_Irecv(C[rankStart+i], N, MPI_INT, rnk, TAG3+i, MPI_COMM_WORLD, &reqsC[rankStart+i-batchSize]);
            }

            // MPI_Status stats[rankBatch];
            // for(int i=0; i<rankBatch; ++i)
            //     MPI_Recv(C[rankStart+i], N, MPI_INT, rnk, TAG3+i, MPI_COMM_WORLD, &stats[i]);
        }

        MPI_Waitall(N-batchSize, reqsC, statsC);
        
        // Stopping the timer:
        end_time = get_usecs();
        double dur = ((double)(end_time-start_time))/1000000;

        // Validating the output
        for(int i=0; i<N; i++) for(int j=0; j<N; j++) assert(C[i][j] == N);
        printf("Test Success. \n");
        printf("Time = %.3f\n", dur);
    }

    // for (int i = 0; i < numRows; i++)
    // {
    //     for (int j = 0; j < N; j++)
    //         printf("%d ", C[i][j]);
    //     printf("| row %d | rank %d \n", i, rank);
    // }
    
    // A[0][0] = 4;
    // B[0][0] = 4;
    // C[0][0] = 4;

    // Free Up the allocated memory space
    // for(int i=0; i<N; ++i) free(A[i]);
	free(A);
    free(B);
    free(C);
    if(rank > 0)
    {
        free(reqsA);
        free(statsA);
    }
    else 
    {
        free(reqsC);
        free(statsC);
    }
    MPI_Finalize();


	return 0;
}

