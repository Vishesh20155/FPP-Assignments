EXE=iterative_average

all: $(EXE)

iterative_average: iterative_average.c
	mpicc -o $@ $^ -fopenmp

clean:
	rm -rf *.o $(EXE)
