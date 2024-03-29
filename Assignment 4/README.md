# Assignment 4

## Question 1:

Steps to run the code:

`
make
`

 `
 OMP_NUM_THREADS=2 mpirun --oversubscribe -np 2 ./iterative_average 25165824 64
 `

 Number of threads can be set by using `OMP_NUM_THREADS` option while `np` option is used to set the number of processors. 

 Last 2 numbers are the length of the array and the number of iterations to be performed.

 ## Question 2:

* `mm_mpi.c` uses Broadcast to send and receive matrices A & B while C is sent using non-blocking calls

* `mm_mpi_blocking.c` uses blocking send and receive for A & C. B is Broadcasted

* `mm_mpi_non_blocking.c` non-blocking send and receive for A & C.

`make`

`mpirun -np 2 ./mm_mpi`

`mpirun -np 2 ./mm_mpi_blocking`

`mpirun -np 2 ./mm_mpi_blocking`


Speedup results:

![image](https://github.com/Vishesh20155/FPP-Assignments/assets/88935135/8ef8e591-2cbb-45c7-af32-e82ad7923007)


