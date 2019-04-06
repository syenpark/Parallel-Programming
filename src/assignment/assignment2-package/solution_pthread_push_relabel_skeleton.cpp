#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <cassert>

#include <vector>
#include <queue>
#include <iostream>

#include "pthread_push_relabel.h"

using namespace std;

void pre_flow(int *dist, int64_t *excess, int *cap, int *flow, int N, int src) {
    dist[src] = N;
    for (auto v = 0; v < N; v++) {
        flow[utils::idx(src, v, N)] = cap[utils::idx(src, v, N)];
        flow[utils::idx(v, src, N)] = -flow[utils::idx(src, v, N)];
        excess[v] = flow[utils::idx(src, v, N)];
    }
}

struct ThreadParameters {
    int num_threads;
    int thread_id;

    int N;
    int src;
    int sink;
    int *cap;
    int *flow;

    int64_t *excess;
    int *dist;

    vector<int> *active_nodes;

    int *single_counter;

    pthread_barrier_t *barrier;
    pthread_mutex_t *single_mtx;
};

void *thread_work(void *t) {
    auto *parameters = (ThreadParameters *) t;
    int num_threads = parameters->num_threads;
    int tid = parameters->thread_id;

    auto src = parameters->src;
    auto sink = parameters->sink;
    int N = parameters->N;
    int *cap = parameters->cap;
    int *flow = parameters->flow;

    int64_t *excess = parameters->excess;
    int *dist = parameters->dist;
    vector<int> &active_nodes = *parameters->active_nodes;

    pthread_barrier_t *barrier = parameters->barrier;
    pthread_mutex_t *single_mtx = parameters->single_mtx;
    int &single_counter = *parameters->single_counter;

    vector<int64_t> tls_stash_excess(N);
    vector<int> tls_stash_dist(N);

    // Four-Stage Pulses.
    while (!active_nodes.empty()) {
        // Stage 1: push.
        int avg = (active_nodes.size() + num_threads - 1) / num_threads;
        int nodes_beg = avg * tid;
        int nodes_end = min<int>(avg * (tid + 1), active_nodes.size());

        for (auto nodes_it = nodes_beg; nodes_it < nodes_end; nodes_it++) {
            auto u = active_nodes[nodes_it];
            for (auto v = 0; v < N; v++) {
                auto residual_cap = cap[utils::idx(u, v, N)] - flow[utils::idx(u, v, N)];
                if (residual_cap > 0 && dist[u] > dist[v] && excess[u] > 0) {
                    auto send = std::min<int64_t>(excess[u], residual_cap);
                    excess[u] -= send;
                    tls_stash_excess[v] += send;
                    flow[utils::idx(u, v, N)] += send;
                    flow[utils::idx(v, u, N)] -= send;
                }
            }
        }
        // Sync flow (reversed edge to be used in stage 2), tls_stash_excess (stage 4).
        pthread_barrier_wait(barrier);

        // Stage 2: relabel (update dist to stash_dist).
        for (auto nodes_it = nodes_beg; nodes_it < nodes_end; nodes_it++) {
            auto u = active_nodes[nodes_it];
            if (excess[u] > 0) {
                int min_dist = INT32_MAX;
                for (auto v = 0; v < N; v++) {
                    auto residual_cap = cap[utils::idx(u, v, N)] - flow[utils::idx(u, v, N)];
                    if (residual_cap > 0) {
                        min_dist = min(min_dist, dist[v]);
                        tls_stash_dist[u] = min_dist + 1;
                    }
                }
            }
        }

        // Sync before update dist (otherwise race condition on dist: stages 2 and 3).
        pthread_barrier_wait(barrier);

        // Stage 3: update dist.
        for (auto nodes_it = nodes_beg; nodes_it < nodes_end; nodes_it++) {
            auto u = active_nodes[nodes_it];
            if (tls_stash_dist[u] > dist[u]) {
                dist[u] = tls_stash_dist[u];
            }
        }

        // Stage 4: apply excess-flow changes for destination vertices.
        for (auto v = 0; v < N; v++) {
            if (tls_stash_excess[v] > 0) {
                // Atomic update excess[v].
                int64_t old_val;
                int64_t new_val;
                do {
                    old_val = excess[v];
                    new_val = old_val + tls_stash_excess[v];
                } while (!__sync_bool_compare_and_swap(&excess[v], old_val, new_val));
                tls_stash_excess[v] = 0;
            }
        }

        // Sync excess[*]
        pthread_barrier_wait(barrier);
        // Single: Construct active nodes.
        pthread_mutex_lock(single_mtx);
        if (single_counter % num_threads == 0) {
            active_nodes.clear();
            for (auto u = 0; u < N; u++) {
                if (excess[u] > 0 && u != src && u != sink) {
                    active_nodes.emplace_back(u);
                }
            }
        }
        single_counter = (single_counter + 1) % num_threads;
        pthread_mutex_unlock(single_mtx);

        // Sync active_nodes
        pthread_barrier_wait(barrier);
    }
    pthread_exit(nullptr);
}

int push_relabel(int num_threads, int N, int src, int sink, int *cap, int *flow) {
    int *dist = (int *) calloc(N, sizeof(int));
    auto *excess = (int64_t *) calloc(N, sizeof(int64_t));
    // PreFlow.
    pre_flow(dist, excess, cap, flow, N, src);

    vector<int> active_nodes;
    for (auto u = 0; u < N; u++) {
        if (u != src && u != sink) {
            active_nodes.emplace_back(u);
        }
    }

    pthread_attr_t attr;
    vector<pthread_t> threads(num_threads);

    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, nullptr, num_threads);

    pthread_mutex_t single_mutex;
    pthread_mutex_init(&single_mutex, nullptr);

    /* For portability, explicitly create threads in a joinable state */
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    int single_counter = 0;
    vector<ThreadParameters> parameters(num_threads);
    ThreadParameters parameters_template{};
    parameters_template.num_threads = num_threads;

    parameters_template.N = N;
    parameters_template.src = src;
    parameters_template.sink = sink;
    parameters_template.cap = cap;
    parameters_template.flow = flow;

    parameters_template.excess = excess;
    parameters_template.dist = dist;
    parameters_template.active_nodes = &active_nodes;

    parameters_template.barrier = &barrier;
    parameters_template.single_counter = &single_counter;
    parameters_template.single_mtx = &single_mutex;

    for (int i = 0; i < num_threads; i++) {
        parameters[i] = parameters_template;
        parameters[i].thread_id = i;
        assert(pthread_create(&threads[i], &attr, thread_work, (void *) &parameters[i]) == 0);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    pthread_barrier_destroy(&barrier);
    pthread_mutex_destroy(&single_mutex);

    free(dist);
    free(excess);

    return 0;
}
