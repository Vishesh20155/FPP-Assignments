#include "stamp.h"			// stamp APIs are declared here
#include <assert.h>

int main(int argc, char** argv) {
  // intialize problem size
  int numThread = argc>1 ? atoi(argv[1]) : 1;
  int size = argc>2 ? atoi(argv[2]) : 48000000;  
  // allocate vectors
  int* A = new int[size];
  int* B = new int[size];
  int* C = new int[size];
  // initialize the vectors
  std::fill(A, A+size, 1);
  std::fill(B, B+size, 1);
  std::fill(C, C+size, 0);
  // start the timer
  // start the parallel addition of two vectors
  stamp::parallel_for(0, size, 1, [&](int i) {
    C[i] = A[i] + B[i];
  }, numThread);
  // end the timer
  // verify the result vector
  for(int i=0; i<size; i++) assert(C[i] == 2);
  printf("Test Success\n");
  // cleanup memory
  delete[] A;
  delete[] B;
  delete[] C;
  return 0;
}

