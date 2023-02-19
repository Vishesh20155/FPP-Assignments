#include <functional>

void find_and_execute_task();
void* worker_routine(void* arg);
// void* pop_task_from_runtime();
std::function<void()> pop_task_from_runtime();