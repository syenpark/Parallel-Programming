/**
 * Name: Seoyoung Park
 * Student id:
 * ITSC email: sparkap@connect.ust.hk
*/
#include <cstring>
#include <cstdint>
#include <cstdlib>

#include <vector>
#include <iostream>

#include "pthread_push_relabel.h"
#include <pthread.h>
#include <semaphore.h>

/*
 *  You can add helper functions and variables as you wish.
 */
using namespace std;

/* Global Variables */
pthread_mutex_t barrier_mutex;
pthread_cond_t ok_to_proceed;

vector<int> active_nodes;
int64_t *excess, *stash_excess;
int counter = 0, *dist, *stash_dist, *stash_send, glob_num_threads, glob_N, glob_src, glob_sink, *glob_cap, *glob_flow;

void pre_flow(int *dist, int64_t *excess, int *cap, int *flow, int N, int src) {
    dist[src] = N;

    for (auto v = 0; v < N; v++) {
        flow[utils::idx(src, v, N)] = cap[utils::idx(src, v, N)];
        flow[utils::idx(v, src, N)] = -flow[utils::idx(src, v, N)];
        excess[v] = flow[utils::idx(src, v, N)];
    }
}
void initialize_globs(int num_threads, int N, int src, int sink, int *cap, int *flow) {
    glob_num_threads = num_threads;
    glob_N = N;
    glob_src = src;
    glob_sink = sink;
    glob_cap = cap;
    glob_flow = flow;
}

void *Hello(void* rank) {
    long my_rank = (long) rank;

    while (!active_nodes.empty()) {
        int avg = (active_nodes.size() + glob_num_threads - 1) / glob_num_threads;
        int nodes_beg = avg * my_rank;
        int nodes_end = min<int>(avg * (my_rank + 1), active_nodes.size());

        // Stage 1: push.
        for (auto nodes_it = nodes_beg; nodes_it < nodes_end; nodes_it++) {
            auto u = active_nodes[nodes_it];

            for (auto v = 0; v < glob_N; v++) {
                auto residual_cap = glob_cap[utils::idx(u, v, glob_N)] - glob_flow[utils::idx(u, v, glob_N)];

                if (residual_cap > 0 && dist[u] > dist[v] && excess[u] > 0) {
                    stash_send[utils::idx(u, v, glob_N)] = std::min<int64_t>(excess[u], residual_cap);

                    if (stash_send[utils::idx(u, v, glob_N)] > 0) {
                        excess[u] -= stash_send[utils::idx(u, v, glob_N)];
                        glob_flow[utils::idx(u, v, glob_N)] += stash_send[utils::idx(u, v, glob_N)];
                        glob_flow[utils::idx(v, u, glob_N)] -= stash_send[utils::idx(u, v, glob_N)];
                        pthread_mutex_lock(&barrier_mutex);
                        stash_excess[v] += stash_send[utils::idx(u, v, glob_N)];
                        pthread_mutex_unlock(&barrier_mutex);
                        stash_send[utils::idx(u, v, glob_N)] = 0;
                    }
                }
            }
        }

        // Stage 2: relabel (update dist to stash_dist).
        memcpy(stash_dist, dist, glob_N * sizeof(int));

        for (auto nodes_it = nodes_beg; nodes_it < nodes_end; nodes_it++) {
            auto u = active_nodes[nodes_it];
            stash_dist[u] = dist[u];

            if (excess[u] > 0) {
                int min_dist = INT32_MAX;

                for (auto v = 0; v < glob_N; v++) {
                    auto residual_cap = glob_cap[utils::idx(u, v, glob_N)] - glob_flow[utils::idx(u, v, glob_N)];

                    if (residual_cap > 0) {
                        min_dist = min(min_dist, dist[v]);
                        stash_dist[u] = min_dist + 1;
                    }
                }
            }
        }

        // Stage 3: update dist.
        //swap(dist, stash_dist);
        for (auto nodes_it = nodes_beg; nodes_it < nodes_end; nodes_it++) {
            auto u = active_nodes[nodes_it];
            dist[u] = stash_dist[u];
        }

        // Barrier
        // Stage 4: apply excess-flow changes for destination vertices.
        pthread_mutex_lock(&barrier_mutex);
        counter++;

        if (counter == glob_num_threads) {
            for (auto v = 0; v < glob_N; v++) {
                if (stash_excess[v] != 0) {
                    excess[v] += stash_excess[v];
                    stash_excess[v] = 0;
                }
            }

            // Construct active nodes.
            active_nodes.clear();

            for (auto u = 0; u < glob_N; u++) {
                if (excess[u] > 0 && u != glob_src && u != glob_sink) {
                    active_nodes.emplace_back(u);
                }
            }

            pthread_cond_broadcast(&ok_to_proceed);
            counter = 0;
        } else {
            while (pthread_cond_wait(&ok_to_proceed, &barrier_mutex) != 0);
        }

        pthread_mutex_unlock(&barrier_mutex);
    }
}

int push_relabel(int num_threads, int N, int src, int sink, int *cap, int *flow) {
    /*
     *  Please fill in your codes here.
     */
    // all things into threds
    long       thread;  /* Use long in case of a 64-bit system */
    pthread_t* thread_handles;

    thread_handles = (pthread_t*) malloc (num_threads*sizeof(pthread_t));

    // sequential
    // two for loop into one for loop

    // syncrhonization between step 1 and 2

    // pthread creat function
    initialize_globs(num_threads, N, src, sink, cap, flow);

    dist = (int *) calloc(N, sizeof(int));
    stash_dist = (int *) calloc(N, sizeof(int));
    excess = (int64_t *) calloc(N, sizeof(int64_t));
    stash_excess = (int64_t *) calloc(N, sizeof(int64_t));

    // PreFlow.
    pre_flow(dist, excess, cap, flow, N, src);

    stash_send = (int *) calloc(N * N, sizeof(int));

    for (auto u = 0; u < N; u++) {
        if (u != src && u != sink) {
            active_nodes.emplace_back(u);
        }
    }

    for (thread = 0; thread < num_threads; thread++)
        pthread_create(&thread_handles[thread], NULL, Hello, (void*) thread);

    // Stopping the threads
    for (thread = 0; thread < num_threads; thread++)
        pthread_join(thread_handles[thread], NULL);

    free(dist);
    free(stash_dist);
    free(excess);
    free(stash_excess);
    free(stash_send);
    free(thread_handles);

    return 0;
}



