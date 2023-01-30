#include "stamp.h"
#include <pthread.h>
#include <iostream>
#include <iomanip>
#include <time.h>

using namespace std;

// Struct for linguistic interface 1(used for fibonacci computation)
// contains the function to be called
typedef struct {
    std::function<void()> call_from_thread;
} interface_1_struct;


// Struct for linguistice interface 2 (used for vector addition)
// This struct can be reused for 3rd type of linguistic interface
typedef struct {
    // Consists of the function to be called, the lower value, high value and the stride for the loop.
    std::function<void(int)> call_from_thread;
    int low, high, stride;
} interface_2_struct;

// Struct for interface 4 and 5 (used for matrix multipication)
typedef struct {
    /*
        Consists of the function to be called, 
        2 low values, 
        2 high values, and 2 stride values for the loop
    */
   
   int low1, low2, high1, high2, stride1, stride2;
   std::function<void(int, int)> call_from_thread;
} interface_4_struct;


// Function called for execution of linguistic interface of first type
// (parallelized Fibonacci function)
void *interface_1_threadFunc(void *args){
    ((interface_1_struct*)args)->call_from_thread();
    return NULL;
}

// Function called for execution of linguistic interface of second type
// (parallelized Vector Addition)
void *interface_2_threadFunc(void *args) {
    interface_2_struct *param = (interface_2_struct*) args;
    for(int i=param->low; i<param->high; i+=param->stride){
        param->call_from_thread(i);
    }

    return NULL;
}

// Function called for execution of linguistic interface of fourth type
// (parallelized Matrix Multiplication)
void *interface_4_threadFunc(void *args) {
    // This function is actually no used, but is for a simpler version of the program which is commented below
    interface_4_struct *param = (interface_4_struct*) args;
    for(int i=param->low1; i<param->high1; i+=param->stride1){
        for(int j=param->low2; j<param->high2; j+=param->stride2){
            param->call_from_thread(i, j);
        }
    }

    return NULL;
}

// This is a function to implement Matrix Multiplication most efficiently in the given scenario
void *interface_4_threadFunc_modified(void *args){
    interface_4_struct *param = (interface_4_struct*) args;
    
    for(int i=param->low1; i<param->high1; ++i){
        int row=i/param->high2, col=i%param->high2;

        param->call_from_thread(row, col);
    }

    return NULL;
}

// Function Definfition for Interface of type 1
void stamp::execute_tuple(std::function<void()> &&lambda1, std::function<void()> &&lambda2) {
    clock_t start, end;
    start = clock();
    pthread_t thread_id;
    interface_1_struct args;
    args.call_from_thread=lambda1;
    int errVal=pthread_create(&thread_id, NULL, interface_1_threadFunc, (void*) &args);
    if(errVal<0){
        perror("Error in pthread_create\n");
        EXIT_FAILURE;
    }

    lambda2();

    errVal=pthread_join(thread_id, NULL);
    if(errVal<0){
        perror("Error in pthread_join\n");
        EXIT_FAILURE;
    }

    end=clock();

    double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
    cout<<"StaMp Statistics: Threads = "<<1<<", ";
    cout<<"Parallel execution time = "<<fixed<< time_taken << setprecision(5);
    cout << " seconds" << endl;
}


// Function Definfition for Interface of type 2
void stamp::parallel_for(int low, int high, int stride, std::function<void(int)> &&lambda, int numThreads) {
    // Incase the size of vector is smaller than numThreads, we should not create extra threads.
    
    clock_t start, end;
    start = clock();

    int threadsCreated=std::min(numThreads, high);
    
    pthread_t thread_ids[threadsCreated];
    interface_2_struct args[threadsCreated];

    // Variable 'extra' accommodates for the fact when size is not a factor of numThreads
    int extra=high%threadsCreated, step=high/threadsCreated, currHigh=step;

    // The loop divides the addition of vectors into addition of size/numThreads sized vectors
    for(int i=0; i<threadsCreated; ++i){
        args[i].call_from_thread=lambda;
        if(i==0){
            args[i].low=0;
        }
        else{
            args[i].low=args[i-1].high;
        }
        args[i].high=currHigh;
        currHigh+=step;
        if(extra){
            extra--;
            args[i].high++;
            currHigh++;
        }
        args[i].stride=stride;

        // Creation of thread
        int errVal=pthread_create(&thread_ids[i], NULL, interface_2_threadFunc, (void*)&(args[i]));
        if(errVal<0){
        perror("Error in pthread_create\n");
        EXIT_FAILURE;
    }
    }

    // for(int i=0; i<numThreads; ++i){
    //     pthread_create(&thread_ids[i], NULL, interface_2_threadFunc, (void*)&(args[i]));
    // }

    for(int i=0; i<threadsCreated; ++i){
        int errVal=pthread_join(thread_ids[i], NULL);
        if(errVal<0){
            perror("Error in pthread_join\n");
            EXIT_FAILURE;
        }
    }

    end=clock();

    double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
    cout<<"StaMp Statistics: Threads = "<<threadsCreated<<", ";
    cout<<"Parallel execution time = "<<fixed<< time_taken << setprecision(5);
    cout << " seconds" << endl;
}

// Function definition for Interface of type 3
void stamp::parallel_for(int high, std::function<void(int)> &&lambda, int numThreads) {
    // Calling the predefined function having similar arguments with low defeaulted to 0 and stride defaulted to 1
    stamp::parallel_for(0, high, 1, (std::function<void(int)>) lambda, numThreads);
}

// Function Definfition for Interface of type 4
void stamp::parallel_for(int low1, int high1, int stride1, int low2, int high2, int stride2, std::function<void(int, int)> &&lambda, int numThreads){
    
    clock_t start, end;
    start=clock();
    
    pthread_t thread_ids[numThreads];
    interface_4_struct args[numThreads];

    // Total no of tasks which is the smallest unit of execution for a thread is size*size
    // This is the number of times the lambda function will be called
    int no_of_tasks=high1*high1;

    int tasks_per_thread=no_of_tasks/numThreads, extra=no_of_tasks%numThreads;

    int idx=0, currHigh=tasks_per_thread, currLow=0;
    while (idx<min(numThreads, no_of_tasks))
    {
        args[idx].call_from_thread=lambda;
        args[idx].low1=currLow;
        if(extra>0){
            currHigh++;
            extra--;
        }
        args[idx].high1=currHigh;

        args[idx].high2=high2;
        
        currLow=currHigh;
        currHigh+=tasks_per_thread;

        int errVal=pthread_create(&thread_ids[idx], NULL, interface_4_threadFunc_modified, (void*)&(args[idx]));
        if(errVal<0){
            perror("Error in pthread_create\n");
            EXIT_FAILURE;
        }

        idx++;
    }

/* 
Below code is commented as it is not the most efficient approach, other approach is implemented above
The logic behind this commented code is explained in further comments and is similar to vector addition function.

    // The comments from this code can be removed and can be run

    // The code written below parallelizes matrix multiplication according to matrix A
    // The matrix A is broken into some rows and then each row group is multiplied with B in the thread function

    // Variable 'extra' accommodates for the fact when size is not a factor of numThreads
    int extra=high1%numThreads, step=high1/numThreads, currHigh=step;

    // The loop divides the addition of vectors into addition of size/numThreads sized vectors
    for(int i=0; i<numThreads; ++i){
        args[i].call_from_thread=lambda;
        args[i].low2=0;
        args[i].high2=high2;
        if(i==0){
            args[i].low1=0;
        }
        else{
            args[i].low1=args[i-1].high1;
        }
        args[i].high1=currHigh;
        currHigh+=step;
        if(extra){
            extra--;
            args[i].high1++;
            currHigh++;
        }
        args[i].stride1=stride1;
        args[i].stride2=stride2;

        // Creation of thread
        pthread_create(&thread_ids[i], NULL, interface_4_threadFunc, (void*)&(args[i]));
    }

*/

    for(int i=0; i<min(numThreads, no_of_tasks); ++i){
        int errVal=pthread_join(thread_ids[i], NULL);
        if(errVal<0){
            perror("Error in pthread_create\n");
            EXIT_FAILURE;
        }
    }

    end=clock();

    double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
    cout<<"StaMp Statistics: Threads = "<<numThreads<<", ";
    cout<<"Parallel execution time = "<<fixed<< time_taken << setprecision(5);
    cout << " seconds" << endl;
}

// Function definition for Interface of type 5
void stamp::parallel_for(int high1, int high2, std::function<void(int, int)> &&lambda, int numThreads) {
    // Calling the predefined function having similar arguments with low1, low2 defaulted to 0 and stride 1, stride 2 defaulted to 1
    stamp::parallel_for(0, high1, 1, 0, high2, 1, (std::function<void(int, int)>) lambda, numThreads);
}
