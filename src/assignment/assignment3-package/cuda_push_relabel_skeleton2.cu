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

#include "cuda_push_relabel.h"

using namespace std;

__device__ int index_d(int x, int y, int n) {
    return x * n + y;
}

__global__ void pre_flow_kernel(int *dist, int64_t *excess, int *cap, int *flow, int N, int src) {
    dist[src] = N;
    const int nthread = blockDim.x * gridDim.x;
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    for (auto v = tid; v < N; v += nthread) {
        flow[index_d(src, v, N)] = cap[index_d(src, v, N)];
        flow[index_d(v, src, N)] = -flow[index_d(src, v, N)];
        excess[v] = flow[index_d(src, v, N)];
    }
}


__global__ void relabel_excess_kernel(int64_t *excess, int *stash_excess, int N){
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int nthread = blockDim.x * gridDim.x;
    for (int v = tid; v < N; v += nthread) {
        if (stash_excess[v] != 0) {
            excess[v] += stash_excess[v];
            stash_excess[v] = 0;
        }
    }
}


__global__ void relabel_kernel(int *dist, int64_t *excess, int *cap, int *flow, int N, int src, int *active_nodes,
                               int count, int *stash_dist) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int nthread = blockDim.x * gridDim.x;

    for (int i = tid; i < count; i += nthread) {
        int u = active_nodes[i];

        if (excess[u] > 0) {
            int min_dist = INT32_MAX;

            for (auto v = 0; v < N; v++) {
                auto residual_cap = cap[index_d(u, v, N)] - flow[index_d(u, v, N)];

                if (residual_cap > 0) {
                    min_dist = min(min_dist, dist[v]);
                    stash_dist[u] = min_dist + 1;
                }
            }
        }
    }
}


__global__ void push_kernel(int *dist, int64_t *excess, int *cap, int *flow, int N, int src, int *active_nodes,
                            int count, int *stash_excess) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int nthread = blockDim.x * gridDim.x;

    for (int i = tid; i < count; i += nthread) {
        int u = active_nodes[i];

        for (auto v = 0; v < N; v++) {
            long residual_cap = cap[index_d(u, v, N)] - flow[index_d(u, v, N)];

            if (residual_cap > 0 && dist[u] > dist[v] && excess[u] > 0) {
                int send_stash = min((long)excess[u], residual_cap);
                excess[u] -= send_stash;

                atomicAdd(flow + index_d(u, v, N), send_stash);
                atomicSub(flow + index_d(v, u, N), send_stash);
                atomicAdd(stash_excess + v, send_stash);
           }
        }
    }
}

int push_relabel(int blocks_per_grid, int threads_per_block, int N, int src, int sink, int *cap, int *flow) {

    int *h_dist = (int *) calloc(N, sizeof(int));
    int *d_dist, *d_stash_dist;
    cudaMalloc(&d_dist, N * sizeof(int));
    cudaMalloc(&d_stash_dist, N * sizeof(int));

    int64_t *excess = (int64_t *) calloc(N, sizeof(int64_t));
    int64_t *d_excess;
    cudaMalloc(&d_excess, N * sizeof(int64_t));

    int *stash_excess = (int *) calloc(N, sizeof(int));
    int *d_stash_excess;
    cudaMalloc(&d_stash_excess, N * sizeof(int));
    
    int *d_cap, *d_flow;
    cudaMalloc(&d_cap, N * N * sizeof(int));
    cudaMalloc(&d_flow, N * N * sizeof(int));
    
	cudaMemcpy(d_cap, cap, sizeof(int) * N * N, cudaMemcpyHostToDevice);

    pre_flow_kernel<<<blocks_per_grid, threads_per_block>>>(d_dist, d_excess, d_cap, d_flow, N, src);

    vector<int> active_nodes;
    int *active_nodes_d;
    cudaMalloc(&active_nodes_d, N * sizeof(int));

    int *stash_send = (int *) calloc(N * N, sizeof(int));
    for (auto u = 0; u < N; u++) {
        if (u != src && u != sink) {
            active_nodes.emplace_back(u);
        }
    }
    
    // Four-Stage Pulses.
    while (!active_nodes.empty()) {
        int count = active_nodes.size();
        cudaMemcpy(active_nodes_d, active_nodes.data(), sizeof(int) * count, cudaMemcpyHostToDevice);
        push_kernel<<<blocks_per_grid, threads_per_block>>>(d_dist, d_excess, d_cap, d_flow, N, src, active_nodes_d, count, d_stash_excess);
        
		cudaMemcpy(d_stash_dist, d_dist, sizeof(int) * N, cudaMemcpyDeviceToDevice);
		relabel_kernel<<<blocks_per_grid, threads_per_block>>>(d_dist, d_excess, d_cap, d_flow, N, src, active_nodes_d, count, d_stash_dist);
		swap(d_dist, d_stash_dist);
        
		relabel_excess_kernel<<<blocks_per_grid, threads_per_block>>>(d_excess, d_stash_excess, N);
		cudaMemcpy(excess, d_excess, sizeof(int64_t) * N, cudaMemcpyDeviceToHost);
        cudaMemcpy(flow, d_flow, sizeof(int) * N * N, cudaMemcpyDeviceToHost);
        
		active_nodes.clear();
        for (auto u = 0; u < N; u++) {
            if (excess[u] > 0 && u != src && u != sink) {
                active_nodes.emplace_back(u);
            }
        }
    }

    free(h_dist);
    free(excess);
    free(stash_send);
    free(stash_excess);

    return 0;
}
