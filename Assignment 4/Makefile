EXE=iterative_average mm_mpi mm_mpi_blocking mm_mpi_non_blocking

all: $(EXE)

iterative_average: iterative_average.c
	mpicc -o $@ $^ -fopenmp

mm_mpi: mm_mpi.c
	mpicc -o $@ $^

mm_mpi_blocking: mm_mpi_blocking.c
	mpicc -o $@ $^

mm_mpi_non_blocking: mm_mpi_non_blocking.c
	mpicc -o $@ $^

clean:
	rm -rf *.o $(EXE)
