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

// Device code
__device__ int idx(int x, int y, int n) {
    return x * n + y;
}

__global__ void push(int *active_nodes, int active_node_size, int N, int *cap, int *flow, int *dist, int64_t *excess, int *stash_send){
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    int numBlock = gridDim.x;
    int numThread = blockDim.x;
    int v, u;
    extern __shared__ unsigned long long residual_cap[];

    for (int i = blockId; i < active_node_size; i+=numBlock){
        u = active_nodes[i];

        for (v = threadId; v < N; v+=numThread){
            residual_cap[v] = cap[idx(u, v, N)] - flow[idx(u, v, N)];
        }
        //__syncthreads();

        if (threadId == 0){
            v = 0;

            while(excess[u]>0 && v < N){
                if (residual_cap[v] > 0 && dist[u] > dist[v]){
                    stash_send[idx(u, v, N)] = min((unsigned long long)excess[u], residual_cap[v]);
                    excess[u] -= stash_send[idx(u, v, N)];
                }
                v++;
            }
        }
        //__syncthreads();
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
    vector<int> h_active_nodes;
    int *h_stash_send = (int *) calloc(N * N, sizeof(int));

    //printf("%d %d\n", sizeof(int), sizeof(int64_t));

    // Allocate Vectors in device memory (GPU)
    int64_t *d_excess, *d_stash_excess;
    int *d_dist, *d_stash_dist, *d_stash_send, *d_cap, *d_flow;

    cudaMalloc((void**) &d_dist, size_in_int);
    cudaMalloc((void**) &d_excess, size_n_int64_t);
    cudaMalloc((void**) &d_stash_dist, size_in_int);
    cudaMalloc((void**) &d_stash_send, size_in_int);
    cudaMalloc((void**) &d_stash_excess, size_n_int64_t);
    cudaMalloc((void**) &d_cap, N * N * sizeof(int));
    cudaMalloc((void**) &d_flow, N * N * sizeof(int));


    // Initialise input data
    // PreFlow.
    pre_flow(h_dist, h_excess, cap, flow, N, src);

    for (auto u = 0; u < N; u++) {
        if (u != src && u != sink) {
            h_active_nodes.emplace_back(u);
        }
    }

    int* d_active_nodes = &h_active_nodes[0];

    cudaMalloc((void**) &d_active_nodes, h_active_nodes.size() * sizeof(int));

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_dist, h_dist, size_in_int, cudaMemcpyHostToDevice);
    cudaMemcpy(d_excess, h_excess, size_n_int64_t, cudaMemcpyHostToDevice);
    cudaMemcpy(d_stash_dist, h_stash_dist, size_in_int, cudaMemcpyHostToDevice);
    cudaMemcpy(d_stash_send, h_stash_send, size_in_int, cudaMemcpyHostToDevice);
    cudaMemcpy(d_stash_excess, h_stash_excess, size_n_int64_t, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cap, cap, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flow, flow, N * N *  sizeof(int), cudaMemcpyHostToDevice);

    // Four-Stage Pulses.
    while (!h_active_nodes.empty()) {
        // Stage 1: push.
        //foo_kernel<<<blocks_per_grid, threads_per_block>>>(d_dist, d_excess);
        cudaMemcpy(d_active_nodes, h_active_nodes.data(), sizeof(int) * h_active_nodes.size(), cudaMemcpyHostToDevice);
        push<<<blocks_per_grid, threads_per_block>>>(d_active_nodes, h_active_nodes.size(), N, d_cap, d_flow, d_dist, d_excess, d_stash_send);
        /*
        for (auto u : h_active_nodes) {
            for (auto v = 0; v < N; v++) {
                auto residual_cap = cap[utils::idx(u, v, N)] - flow[utils::idx(u, v, N)];

                if (residual_cap > 0 && h_dist[u] > h_dist[v] && h_excess[u] > 0) {
                    h_stash_send[utils::idx(u, v, N)] = std::min<int64_t>(h_excess[u], residual_cap);
                    h_excess[u] -= h_stash_send[utils::idx(u, v, N)];
                }

                if (h_stash_send[utils::idx(u, v, N)] > 0) {
                    flow[utils::idx(u, v, N)] += h_stash_send[utils::idx(u, v, N)];
                    flow[utils::idx(v, u, N)] -= h_stash_send[utils::idx(u, v, N)];
                    h_stash_excess[v] += h_stash_send[utils::idx(u, v, N)];
                    h_stash_send[utils::idx(u, v, N)] = 0;
                }
            }
        }
        */

        // Copy result
        //cudaMemcpy(h_active_nodes, d_active_nodes, size_in_int, cudaMemcpyDeviceToHost);
        cudaMemcpy(cap, d_cap, size_in_int, cudaMemcpyDeviceToHost);
        cudaMemcpy(flow, d_flow, size_in_int, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_dist, d_dist, size_in_int, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_excess, d_excess, size_in_int, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_stash_send, d_stash_send, size_in_int, cudaMemcpyDeviceToHost);

        // Stage 2: relabel (update dist to stash_dist).
        memcpy(h_stash_dist, h_dist, N * sizeof(int));
        for (auto u : h_active_nodes) {
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
        h_active_nodes.clear();
        for (auto u = 0; u < N; u++) {
            if (h_excess[u] > 0 && u != src && u != sink) {
                h_active_nodes.emplace_back(u);
            }
        }
    }

    // Free host memory
    free(h_dist);
    free(h_excess);
    free(h_stash_send);
    free(h_stash_dist);
    free(h_stash_excess);

    // Free device memory
    cudaFree(d_dist);
    cudaFree(d_excess);
    cudaFree(d_stash_dist);
    cudaFree(d_stash_send);
    cudaFree(d_stash_excess);

    return 0;
}
