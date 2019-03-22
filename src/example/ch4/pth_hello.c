/* File:  
 *    pth_hello.c
 *
 * Purpose:
 *    Illustrate basic use of pthreads:  create some threads,
 *    each of which prints a message.
 *
 * Input:
 *    none
 * Output:
 *    message from each thread
 *
 * Compile:  gcc -g -Wall -o pth_hello pth_hello.c -lpthread
 * Usage:    ./pth_hello <thread_count>
 *
 * IPP:   Section 4.2 (p. 153 and ff.)
 */
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h> 

const int MAX_THREADS = 64;

/* Global variable:  accessible to all threads */
int thread_count;  

void Usage(char* prog_name);
void *Hello(void* rank);  /* Thread function */

/*--------------------------------------------------------------------*/
int main(int argc, char* argv[]) {
   long       thread;  /* Use long in case of a 64-bit system */
   pthread_t* thread_handles;

   // If not <program_name> <the # of threads> format, print the alert message
   if (argc != 2) Usage(argv[0]);

   /* Get number of threads from command line */
   thread_count = strtol(argv[1], NULL, 10);

   // If the number of threads is not proper, print the alert message
   if (thread_count <= 0 || thread_count > MAX_THREADS) Usage(argv[0]);

   thread_handles = malloc (thread_count*sizeof(pthread_t));

   /*
    * pthread.h -> pthread_t: One object for each thread
    * int pthread_create(pthread_t* thread_p, const pthread_attr_t* attr_p, void, void);
    *                         out                        in                  in    in
    *
    * pthread_t* thread_p should be allocated before calling by above code
    * const pthread_attr_t* attr_p; We won't be using, so we just pass NULL
    * The function that the read is to run
    * Pointer to the argument that should be passed to the function start_routine
    */

   for (thread = 0; thread < thread_count; thread++)  
      pthread_create(&thread_handles[thread], NULL, Hello, (void*) thread);

   printf("Hello from the main thread\n");

   // Stopping the threads
   for (thread = 0; thread < thread_count; thread++) 
      pthread_join(thread_handles[thread], NULL); 

   free(thread_handles);

   return 0;
}  /* main */

/*-------------------------------------------------------------------*/
void *Hello(void* rank) {
   /*
    * Print greeting message per each thread
    */
   long my_rank = (long) rank;  /* Use long in case of 64-bit system */ 

   printf("Hello from thread %ld of %d\n", my_rank, thread_count);

   return NULL;
}  /* Hello */

/*-------------------------------------------------------------------*/
void Usage(char* prog_name) {
   /*
    * Print alert message about invalid input (argument style or the number of it)
    */
   fprintf(stderr, "usage: %s <number of threads>\n", prog_name);
   fprintf(stderr, "0 < number of threads <= %d\n", MAX_THREADS);
   exit(0);
}  /* Usage */
