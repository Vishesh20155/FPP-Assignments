#include "stamp.h"
#include <pthread.h>

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


void stamp::execute_tuple(std::function<void()> &&lambda1, std::function<void()> &&lambda2) {
    pthread_t thread_id;
    interface_1_struct args;
    args.call_from_thread=lambda1;
    pthread_create(&thread_id, NULL, interface_1_threadFunc, (void*) &args);
    lambda2();
    pthread_join(thread_id, NULL);
}

void stamp::parallel_for(int low, int high, int stride, std::function<void(int)> &&lambda, int numThreads) {
    pthread_t thread_ids[numThreads];
    interface_2_struct args[numThreads];

    // Variable 'extra' accommodates for the fact when size is not a factor of numThreads
    int extra=high%numThreads, step=high/numThreads, currHigh=step;

    // The loop divides the addition of vectors into addition of size/numThreads sized vectors
    for(int i=0; i<numThreads; ++i){
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
        pthread_create(&thread_ids[i], NULL, interface_2_threadFunc, (void*)&(args[i]));
    }

    // for(int i=0; i<numThreads; ++i){
    //     pthread_create(&thread_ids[i], NULL, interface_2_threadFunc, (void*)&(args[i]));
    // }

    for(int i=0; i<numThreads; ++i){
        pthread_join(thread_ids[i], NULL);
    }
}