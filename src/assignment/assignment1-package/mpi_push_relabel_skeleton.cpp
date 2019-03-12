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

#include "mpi_push_relabel.h"

using namespace std;

void pre_flow(int *dist, int64_t *excess, int *cap, int *flow, int N, int src) {
    dist[src] = N;

    for (auto v = 0; v < N; v++) {
        flow[utils::idx(src, v, N)] = cap[utils::idx(src, v, N)];
        flow[utils::idx(v, src, N)] = -flow[utils::idx(src, v, N)];
        excess[v] = flow[utils::idx(src, v, N)];
    }
}

int push_relabel(int my_rank, int p, MPI_Comm comm, int N, int src, int sink, int *cap, int *flow) {
    int *dist = (int *) calloc(N, sizeof(int));
    int *stash_dist = (int *) calloc(N, sizeof(int));
    auto *excess = (int64_t *) calloc(N, sizeof(int64_t));
    auto *stash_excess = (int64_t *) calloc(N, sizeof(int64_t));

    // PreFlow.
    pre_flow(dist, excess, cap, flow, N, src);

    // Broadcast
    int* local_N = (int *) calloc(1, sizeof(int));
    int* local_src = (int *) calloc(1, sizeof(int));
    int* local_sink = (int *) calloc(1, sizeof(int));
    int* local_cap = (int *) calloc(N, sizeof(int));
    int* local_flow = (int *) calloc(N, sizeof(int));

    if (my_rank == 0) {
        *local_N = N;
        *local_src = src;
        *local_sink = sink;
        memcpy(local_cap, cap, sizeof(int)*N);
        memcpy(local_flow, flow, sizeof(int)*N);
    }

    MPI_Bcast(local_N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(local_src, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(local_sink, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(local_cap, N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(local_flow, N, MPI_INT, 0, MPI_COMM_WORLD);
    fprintf(stderr, "ranl %d N %d Sink %d %d\n\n", my_rank, *local_N, *local_sink, *local_cap);

    // for all the procedure
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
                auto residual_cap = cap[utils::idx(u, v, N)] - flow[utils::idx(u, v, N)];

                if (residual_cap > 0 && dist[u] > dist[v] && excess[u] > 0) {
                    stash_send[utils::idx(u, v, N)] = std::min<int64_t>(excess[u], residual_cap);
                    excess[u] -= stash_send[utils::idx(u, v, N)]; //
                }
            }
        }
        for (auto u : active_nodes) {
            for (auto v = 0; v < N; v++) {
                if (stash_send[utils::idx(u, v, N)] > 0) {
                    flow[utils::idx(u, v, N)] += stash_send[utils::idx(u, v, N)];
                    flow[utils::idx(v, u, N)] -= stash_send[utils::idx(u, v, N)];
                    stash_excess[v] += stash_send[utils::idx(u, v, N)];
                    stash_send[utils::idx(u, v, N)] = 0;
                }
            }
        }

        // Stage 2: relabel (update dist to stash_dist).
        memcpy(stash_dist, dist, N * sizeof(int));
        for (auto u : active_nodes) {
            if (excess[u] > 0) {
                int min_dist = INT32_MAX;
                for (auto v = 0; v < N; v++) {
                    auto residual_cap = cap[utils::idx(u, v, N)] - flow[utils::idx(u, v, N)];
                    if (residual_cap > 0) {
                        min_dist = min(min_dist, dist[v]);
                        stash_dist[u] = min_dist + 1;
                    }
                }
            }
        }

        // Stage 3: update dist.
        swap(dist, stash_dist);

        // Stage 4: apply excess-flow changes for destination vertices.
        for (auto v = 0; v < N; v++) {
            if (stash_excess[v] != 0) {
                excess[v] += stash_excess[v];
                stash_excess[v] = 0;
            }
        }

        // Construct active nodes.
        active_nodes.clear();
        for (auto u = 0; u < N; u++) {
            if (excess[u] > 0 && u != src && u != sink) {
                active_nodes.emplace_back(u);
            }
        }
    }

    free(dist);
    free(stash_dist);
    free(excess);
    free(stash_excess);
    free(stash_send);

    free(local_N);
    free(local_src);
    free(local_sink);
    free(local_cap);
    free(local_flow);

    return 0;
}