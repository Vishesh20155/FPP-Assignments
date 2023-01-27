#include "stamp.h"			// stamp APIs are declared here
int fib(int n) {
    if(n<2) return n;
    else return fib(n-1) + fib(n-2);
}
int main(int argc, char** argv) {
    int n = 40;
    int result;
    if(n<30) result = fib(n);
    else {
        int x, y;
        // stamp API to execute two tasks (C++11 lambda) in parallel
        stamp::execute_tuple([&]() { x=fib(n-1); }, [&]() { y=fib(n-2); } );
        result = x + y;
    }
    printf("Fib of %d = %d\n",n, result);
    return 0;
}
