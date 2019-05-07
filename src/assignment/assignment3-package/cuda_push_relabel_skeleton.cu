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

#include "cuda_push_relabel.h"

using namespace std;

void pre_flow(int *dist, int64_t *excess, int *cap, int *flow, int N, int src) {
    dist[src] = N;

    for (auto v = 0; v < N; v++) {
        flow[utils::idx(src, v, N)] = cap[utils::idx(src, v, N)];
        flow[utils::idx(v, src, N)] = -flow[utils::idx(src, v, N)];
        excess[v] = flow[utils::idx(src, v, N)];
    }
}

int push_relabel(int blocks_per_grid, int threads_per_block, int N, int src, int sink, int *cap, int *flow) {
    size_t size_in_int = N * sizeof(int);
    size_t size_n_int64_t = N * sizeof(int64_t);

    // Allocate Vectors in host memory (CPU)
    int *h_dist = (int *) calloc(N, sizeof(int));
    int *h_stash_dist = (int *) calloc(N, sizeof(int));
    auto *h_excess = (int64_t *) calloc(N, sizeof(int64_t));
    auto *h_stash_excess = (int64_t *) calloc(N, sizeof(int64_t));

    //printf("%d %d\n", sizeof(int), sizeof(int64_t));
    /*
    // Allocate Vectors in device memory (GPU)
    int *d_dist = (int *) calloc(N, sizeof(int));
    int *d_stash_dist = (int *) calloc(N, sizeof(int));
    auto *d_excess = (int64_t *) calloc(N, sizeof(int64_t));
    auto *d_stash_excess = (int64_t *) calloc(N, sizeof(int64_t));
    int *d_stash_send = (int *) calloc(N * N, sizeof(int));

    cudaMalloc((void**) &d_dist, size);
    cudaMalloc((void**) &d_stash_dist, size);
    cudaMalloc((void**) &d_excess, size);
    cudaMalloc((void**) &d_stash_excess, size);
    cudaMalloc((void**) &d_stash_send, size);
     */


    // PreFlow.
    pre_flow(h_dist, h_excess, cap, flow, N, src);

    vector<int> active_nodes;
    int *stash_send = (int *) calloc(N * N, sizeof(int));

    for (auto u = 0; u < N; u++) {
        if (u != src && u != sink) {
            active_nodes.emplace_back(u);
        }
    }

    // Four-Stage Pulses.
    while (!active_nodes.empty()) {
        // Stage 1: push.
        for (auto u : active_nodes) {
            for (auto v = 0; v < N; v++) {
                auto residual_cap = cap[utils::idx(u, v, N)] -
                                    flow[utils::idx(u, v, N)];
                if (residual_cap > 0 && h_dist[u] > h_dist[v] && h_excess[u] > 0) {
                    stash_send[utils::idx(u, v, N)] = std::min<int64_t>(h_excess[u], residual_cap);
                    h_excess[u] -= stash_send[utils::idx(u, v, N)];
                }
            }
        }
        for (auto u : active_nodes) {
            for (auto v = 0; v < N; v++) {
                if (stash_send[utils::idx(u, v, N)] > 0) {
                    flow[utils::idx(u, v, N)] += stash_send[utils::idx(u, v, N)];
                    flow[utils::idx(v, u, N)] -= stash_send[utils::idx(u, v, N)];
                    h_stash_excess[v] += stash_send[utils::idx(u, v, N)];
                    stash_send[utils::idx(u, v, N)] = 0;
                }
            }
        }

        // Stage 2: relabel (update dist to stash_dist).
        memcpy(h_stash_dist, h_dist, N * sizeof(int));
        for (auto u : active_nodes) {
            if (h_excess[u] > 0) {
                int min_dist = INT32_MAX;
                for (auto v = 0; v < N; v++) {
                    auto residual_cap = cap[utils::idx(u, v, N)] - flow[utils::idx(u, v, N)];
                    if (residual_cap > 0) {
                        min_dist = min(min_dist, h_dist[v]);
                        h_stash_dist[u] = min_dist + 1;
                    }
                }
            }
        }

        // Stage 3: update dist.
        swap(h_dist, h_stash_dist);

        // Stage 4: apply excess-flow changes for destination vertices.
        for (auto v = 0; v < N; v++) {
            if (h_stash_excess[v] != 0) {
                h_excess[v] += h_stash_excess[v];
                h_stash_excess[v] = 0;
            }
        }

        // Construct active nodes.
        active_nodes.clear();
        for (auto u = 0; u < N; u++) {
            if (h_excess[u] > 0 && u != src && u != sink) {
                active_nodes.emplace_back(u);
            }
        }
    }

    free(h_dist);
    free(h_stash_dist);
    free(h_excess);
    free(h_stash_excess);
    free(stash_send);

    return 0;
}
