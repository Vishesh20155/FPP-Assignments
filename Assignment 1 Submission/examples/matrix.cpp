#include "stamp.h"			// stamp APIs are declared here
#include <assert.h>

int main(int argc, char** argv) {
  // intialize problem size
  int numThread = argc>1 ? atoi(argv[1]) : 1;
  int size = argc>2 ? atoi(argv[2]) : 1024;  
  // allocate matrices
  int** A = new int*[size];
  int** B = new int*[size];
  int** C = new int*[size];
  for(int i=0; i<size; i++) {
    A[i] = new int[size];
    B[i] = new int[size];
    C[i] = new int[size];
    for(int j=0; j<size; j++) {
      // initialize the matrices
      std::fill(A[i], A[i]+size, 1);
      std::fill(B[i], B[i]+size, 1);
      std::fill(C[i], C[i]+size, 0);
    }
  }  
  // start the parallel multiplication of two matrices
  stamp::parallel_for(0, size, 1, 0, size, 1, [&](int i, int j) {
    for(int k=0; k<size; k++) {
      C[i][j] += A[i][k] * B[k][j];
    }
  }, numThread);
  // verify the result matrix
  for(int i=0; i<size; i++) for(int j=0; j<size; j++) assert(C[i][j] == size);
  printf("Test Success. \n");
  // cleanup memory
  for(int i=0; i<size; i++) {
    delete A[i];
    delete B[i];
    delete C[i];
  }
  delete[] A;
  delete[] B;
  delete[] C;
  return 0;
}
