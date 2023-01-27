#ifndef TESTLIB_H
#define TESTLIB_H
#include<functional>

class stamp{
    public:
        // Linguistic Interface 1
        static void execute_tuple(std::function<void()> &&lambda1, std::function<void()> &&lambda2);
        
        // Linguistic Interface 2
        static void parallel_for(int low, int high, int stride, std::function<void(int)> &&lambda, int numThreads);
        
        // Linguistic Interface 3
        static void parallel_for(int high, std::function<void(int)> &&lambda, int numThreads);
        
        // Linguistic Interface 4
        static void parallel_for(int low1, int high1, int stride1, int low2, int high2, int stride2, std::function<void(int, int)> &&lambda, int numThreads);
        
        // Linguistic Interface 5
        static void parallel_for(int high1, int high2, std::function<void(int, int)> &&lambda, int numThreads);

};
#endif