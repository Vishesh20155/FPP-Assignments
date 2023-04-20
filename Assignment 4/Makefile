EXE=iterative_average mm_mpi

all: $(EXE)

iterative_average: iterative_average.c
	mpicc -o $@ $^ -fopenmp

mm_mpi: mm_mpi.c
	mpicc -o $@ $^

clean:
	rm -rf *.o $(EXE)
