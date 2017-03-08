/*
 * Simple solution to Dining Philosophers with pthread
 * Author: CHANG GAO
 * Development platform: gcc (Ubuntu 6.2.0-3ubuntu11~14.04) 6.2.0
 * Last modified date: 29 Jan 2017
 * Compilation: gcc DPP.c -o DPP -Wall -pthread
                ./DPP <Number of philosophers>
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>

/*Global variables */
int num_threads; 
pthread_mutex_t *mutexes;

/* For representing the status of each philosopher */
typedef enum{
	none,   // No forks
	one,    // One fork
	two     // Both forks to consume
} utensil;

/* Representation of a philosopher */
typedef struct phil_data{
	int phil_num;
	int course;
	utensil forks; 
}phil_data;


void *eat_meal(void *param){
	/* 3 course meal: Each need to acquire both forks 3 times.
	 *  First try for fork in front.
	 * Then for the one on the right, if not fetched, put the first one back.
	 * If both acquired, eat one course.
	 */
	phil_data* phi = (phil_data*)param;
	int first_fork = phi->phil_num;
  	int second_fork = (phi->phil_num+1)%num_threads;
  	while (phi->course < 3){
  		// cannot grab fork in the front
  		if (pthread_mutex_trylock(mutexes+first_fork)) continue;
  		phi->forks = one;
	    // cannot grab fork on the right
	    if(pthread_mutex_trylock(mutexes+second_fork)){
	      phi->forks = none;
	      pthread_mutex_unlock(mutexes+first_fork);
	      continue;
	    }
	    // successfully grab two forks
	    phi->forks = two;
	    ++phi->course;
	    fprintf(stdout, "Phil num %2d start course %d\n", phi->phil_num, phi->course);
	    // sleep for one second
	    sleep(1);
	    // finish eating
	    pthread_mutex_unlock(mutexes+first_fork);
	    phi->forks = one;
	    pthread_mutex_unlock(mutexes+second_fork);
	    phi->forks = none;
  	}

  	pthread_exit(NULL);
}


int main(int argc, char **argv){
	int i;
	if (argc < 2) {
        fprintf(stderr, "Format: %s <Number of philosophers>\n", (char*)argv[0]);
        return 0;
    }
    num_threads = atoi(argv[1]);
	pthread_t threads[num_threads];
	phil_data *philosophers = malloc(sizeof(phil_data)*num_threads); //Struct for each philosopher
	mutexes = malloc(sizeof(pthread_mutex_t)*num_threads); //Each mutex element represent a fork
	
	/* Initialize structs */
	for(i = 0; i < num_threads; ++i){
		philosophers[i].phil_num = i;
		philosophers[i].course   = 0;
		philosophers[i].forks    = none;
	}

	/* Initialize Mutex, Create threads, Join threads and Destroy mutex */
	for(i = 0; i < num_threads; ++i) pthread_mutex_init(mutexes+i, NULL);
	for(i = 0; i < num_threads; ++i) pthread_create(&threads[i], NULL, eat_meal, (void *)(philosophers+i));
  	for(i = 0; i < num_threads; ++i) pthread_join(threads[i], NULL);
  	for(i = 0; i < num_threads; ++i) pthread_mutex_destroy(mutexes+i);

	return 0;
}