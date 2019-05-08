/**
 * Name: Seoyoung Park
 * Student id:
 * ITSC email: sparkap@connect.ust.hk
 */
// #define int int64_t
#include <cstring>
#include <cstdint>
#include <cstdlib>

#include <vector>
#include <iostream>

#include "cuda_push_relabel.h"

using namespace std;

__device__ int idx(int x, int y, int n) {
    return x * n + y;
}

__global__ void pre_flow(int *dist, unsigned long long int *excess, int *cap, int *flow, int N, int src) {
    dist[src] = N;
    int num_threads = blockDim.x * gridDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    for (auto v = tid; v < N; v += num_threads) {
        flow[idx(src, v, N)] = cap[idx(src, v, N)];
        flow[idx(v, src, N)] = -flow[idx(src, v, N)];
        excess[v] = flow[idx(src, v, N)];
    }
}

__global__ void push(int *dist, unsigned long long int *excess, int *cap, int *flow, int N, int src, int *active_nodes, int count, unsigned long long int *stash_excess) {
    int num_threads = blockDim.x * gridDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = tid; i < count; i += num_threads) {
        int u = active_nodes[i];

        for (auto v = 0; v < N; v++) {
            long long int residual_cap = cap[idx(u, v, N)] - flow[idx(u, v, N)];

            if (residual_cap > 0 && dist[u] > dist[v] && excess[u] > 0) {
                unsigned long long int tmp = min(excess[u], residual_cap);
                excess[u] -= tmp;
                atomicAdd(flow + idx(u, v, N), tmp);
                atomicSub(flow + idx(v, u, N), tmp);
                atomicAdd(stash_excess + v, tmp);
            }
        }
    }
}


__global__ void relabel(int N, int src, int *dist, unsigned long long int *excess, int *cap, int *flow, int *active_nodes, int count, int *stash_dist) {
    int start = blockDim.x * blockIdx.x + threadIdx.x;
    int elementSkip = blockDim.x * gridDim.x;

    for (int i = start; i < count; i += elementSkip) {
        int u = active_nodes[i];

        if (excess[u] > 0) {
            int min_dist = INT32_MAX;

            for (auto v = 0; v < N; v++) {
                auto residual_cap = cap[idx(u, v, N)] - flow[idx(u, v, N)];
                if (residual_cap > 0) {
                    min_dist = atomicMin(dist + v, min_dist);
                    stash_dist[u] = min_dist + 1;
                }
            }
        }
    }
}

__global__ void apply_changes(int N, unsigned long long int *excess, unsigned long long int *stash_excess){
    int start = blockDim.x * blockIdx.x + threadIdx.x;
    int elementSkip = blockDim.x * gridDim.x;
    for (int v = start; v < N; v += elementSkip) {
        if (stash_excess[v] != 0) {
            excess[v] += stash_excess[v];
            stash_excess[v] = 0;
        }
    }
}

int push_relabel(int blocks_per_grid, int threads_per_block, int N, int src, int sink, int *cap, int *flow) {
    int *dist = (int *) calloc(N, sizeof(int));
    int *stash_dist = (int *) calloc(N, sizeof(int));
    unsigned long long int *excess = (unsigned long long int *) calloc(N, sizeof(unsigned long long));
    unsigned long long int *stash_excess = (unsigned long long int *) calloc(N, sizeof(unsigned long long));

    int *dist_d, *stash_dist_d;
    cudaMalloc(&dist_d, N * sizeof(int));
    cudaMalloc(&stash_dist_d, N * sizeof(int));

    unsigned long long int *excess_d, *stash_excess_d;
    cudaMalloc(&excess_d, N * sizeof(unsigned long long));
    cudaMalloc(&stash_excess_d, N * sizeof(unsigned long long));

    int *cap_d, *flow_d;
    cudaMalloc(&cap_d, N * N * sizeof(int));
    cudaMalloc(&flow_d, N * N * sizeof(int));

    cudaMemcpy(cap_d, cap, sizeof(int) * N * N, cudaMemcpyHostToDevice);
    pre_flow<<<blocks_per_grid, threads_per_block>>>(dist_d, excess_d, cap_d, flow_d, N, src);

    vector<int> active_nodes;
    int *active_nodes_d;
    cudaMalloc(&active_nodes_d, N * sizeof(int));

    unsigned long long int *stash_send = (unsigned long long int *) calloc(N * N, sizeof(unsigned long long));
    for (auto u = 0; u < N; u++) {
        if (u != src && u != sink) {
            active_nodes.emplace_back(u);
        }
    }

    while (!active_nodes.empty()) {
        int count = active_nodes.size();

        cudaMemcpy(active_nodes_d, active_nodes.data(), sizeof(int) * count, cudaMemcpyHostToDevice);
        push<<<blocks_per_grid, threads_per_block>>>(dist_d, excess_d, cap_d, flow_d, N, src, active_nodes_d, count, stash_excess_d);

        cudaMemcpy(stash_dist_d, dist_d, N * sizeof(int), cudaMemcpyDeviceToDevice);
        relabel<<<blocks_per_grid, threads_per_block>>>(N, src, dist_d, excess_d, cap_d, flow_d, active_nodes_d, count, stash_dist_d);

        // Stage 3. Update
        swap(dist_d, stash_dist_d);

        // Stage 4: apply excess-flow changes for destination vertices.
        apply_changes<<<blocks_per_grid, threads_per_block>>>(N, excess_d, stash_excess_d);
        cudaMemcpy(excess, excess_d, sizeof(unsigned long long) * N, cudaMemcpyDeviceToHost);
        cudaMemcpy(flow, flow_d, sizeof(int) * N * N, cudaMemcpyDeviceToHost);

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

    // cudaFree(dist_d);
    // cudaFree(stash_dist_d);
    // cudaFree(excess_d);
    // cudaFree(stash_excess_d);
    // cudaFree(cap_d);
    // cudaFree(flow_d);
    // cudaFree(active_nodes_d);

    return 0;
}
