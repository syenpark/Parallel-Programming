/**
 * Name:
 * Student id:
 * ITSC email:
*/

#include <cstring>
#include <cstdint>
#include <cstdlib>

#include <vector>
#include <iostream>

#include "mpi_push_relabel.h"

using namespace std;

void pre_flow(int *dist, int64_t *excess, int *cap, int *flow, int loc_n, int src) {
    dist[src] = loc_n;
    for (auto v = 0; v < loc_n; v++) {
        flow[utils::idx(src, v, loc_n)] = cap[utils::idx(src, v, loc_n)];
        flow[utils::idx(v, src, loc_n)] = -flow[utils::idx(src, v, loc_n)];
        excess[v] = flow[utils::idx(src, v, loc_n)];
    }
}

#define ROOT (0)

int push_relabel(int my_rank, int p, MPI_Comm comm, int N, int src, int sink, int *cap, int *flow) {
    // Broadcast loc_n.
    int loc_n;
    int loc_src;
    int loc_sink;
    if (my_rank == ROOT) {
        loc_n = N;
        loc_src = src;
        loc_sink = sink;
    }
    MPI_Bcast(&loc_n, 1, MPI_INT, ROOT, comm);
    MPI_Bcast(&loc_src, 1, MPI_INT, ROOT, comm);
    MPI_Bcast(&loc_sink, 1, MPI_INT, ROOT, comm);

    int *dist = (int *) calloc(loc_n, sizeof(int));
    int *stash_dist = (int *) calloc(loc_n, sizeof(int));
    auto *excess = (int64_t *) calloc(loc_n, sizeof(int64_t));
    auto *stash_excess = (int64_t *) calloc(loc_n, sizeof(int64_t));

    int *loc_cap = (int *) malloc(sizeof(int) * loc_n * loc_n);
    int *loc_flow = (int *) malloc(sizeof(int) * loc_n * loc_n);
    int *stash_send = (int *) calloc(loc_n * loc_n, sizeof(int));

    // Broadcast loc_cap, loc_flow
    if (my_rank == ROOT) {
        memcpy(loc_cap, cap, sizeof(int) * loc_n * loc_n);
        memcpy(loc_flow, flow, sizeof(int) * loc_n * loc_n);
    }
    MPI_Bcast(loc_cap, loc_n * loc_n, MPI_INT, ROOT, comm);
    MPI_Bcast(loc_flow, loc_n * loc_n, MPI_INT, ROOT, comm);

    // PreFlow.
    pre_flow(dist, excess, loc_cap, loc_flow, loc_n, src);

    vector<int> active_nodes;
    for (auto u = 0; u < loc_n; u++) {
        if (u != loc_src && u != loc_sink) {
            active_nodes.emplace_back(u);
        }
    }

    // Four-Stage Pulses.
    while (!active_nodes.empty()) {
        // Stage 1: push.
        int avg = (active_nodes.size() + p - 1) / p;
        int nodes_beg = avg * my_rank;
        int nodes_end = min<int>(avg * (my_rank + 1), active_nodes.size());

        for (auto nodes_it = nodes_beg; nodes_it < nodes_end; nodes_it++) {
            auto u = active_nodes[nodes_it];
            for (auto v = 0; v < loc_n; v++) {
                auto residual_cap = loc_cap[utils::idx(u, v, loc_n)] -
                                    loc_flow[utils::idx(u, v, loc_n)];
                if (residual_cap > 0 && dist[u] > dist[v] && excess[u] > 0) {
                    stash_send[utils::idx(u, v, loc_n)] = std::min<int64_t>(excess[u], residual_cap);
                    excess[u] -= stash_send[utils::idx(u, v, loc_n)];
                }
            }
        }
//        for (auto u:active_nodes) {
//            MPI_Allreduce(MPI_IN_PLACE, stash_send + utils::idx(u, 0, loc_n), loc_n, MPI_INT, MPI_MAX, comm);
//        }

        for (auto i = 0; i < p; i++) {
            for (auto nodes_it = avg * i; nodes_it < min<int>(avg * (i + 1), active_nodes.size()); nodes_it++) {
                auto u = active_nodes[nodes_it];
                MPI_Bcast(stash_send + utils::idx(u, 0, loc_n), loc_n, MPI_INT, i, comm);
            }
        }

        for (auto nodes_it = 0; nodes_it < active_nodes.size(); nodes_it++) {
            auto u = active_nodes[nodes_it];
            bool needs_update = !(nodes_it >= nodes_beg && nodes_it < nodes_end);
            for (auto v = 0; v < loc_n; v++) {
                if (stash_send[utils::idx(u, v, loc_n)] > 0) {
                    if (needs_update) {
                        excess[u] -= stash_send[utils::idx(u, v, loc_n)];
                    }
                    loc_flow[utils::idx(u, v, loc_n)] += stash_send[utils::idx(u, v, loc_n)];
                    loc_flow[utils::idx(v, u, loc_n)] -= stash_send[utils::idx(u, v, loc_n)];
                    stash_excess[v] += stash_send[utils::idx(u, v, loc_n)];
                    stash_send[utils::idx(u, v, loc_n)] = 0;
                }
            }
        }

        // Stage 2: relabel (update dist to stash_dist).
        memcpy(stash_dist, dist, loc_n * sizeof(int));
        for (auto nodes_it = nodes_beg; nodes_it < nodes_end; nodes_it++) {
            auto u = active_nodes[nodes_it];
            if (excess[u] > 0) {
                int min_dist = INT32_MAX;
                for (auto v = 0; v < loc_n; v++) {
                    auto residual_cap = loc_cap[utils::idx(u, v, loc_n)] - loc_flow[utils::idx(u, v, loc_n)];
                    if (residual_cap > 0) {
                        min_dist = min(min_dist, dist[v]);
                        stash_dist[u] = min_dist + 1;
                    }
                }
            }
        }
        // Stage 3: update dist.
        MPI_Allreduce(stash_dist, dist, loc_n, MPI_INT, MPI_MAX, comm);

        // Stage 4: apply excess-flow changes for destination vertices.
        for (auto v = 0; v < loc_n; v++) {
            if (stash_excess[v] != 0) {
                excess[v] += stash_excess[v];
                stash_excess[v] = 0;
            }
        }

        // Construct active nodes.
        active_nodes.clear();
        for (auto u = 0; u < loc_n; u++) {
            if (excess[u] > 0 && u != loc_src && u != loc_sink) {
                active_nodes.emplace_back(u);
            }
        }
        MPI_Barrier(comm);
    }

    free(dist);
    free(stash_dist);
    free(excess);
    free(stash_excess);
    free(stash_send);

    if (my_rank == ROOT) {
        memcpy(flow, loc_flow, sizeof(int) * loc_n * loc_n);
    }

    free(loc_cap);
    free(loc_flow);

    return 0;
}
