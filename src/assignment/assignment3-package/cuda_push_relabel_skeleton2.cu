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

__global__ void pre_flow_d(int *dist, int64_t *excess, int *cap, int *flow, int N, int src) {
    dist[src] = N;
    const int nthread = blockDim.x * gridDim.x;
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (auto v = tid; v < N; v += nthread) {
        flow[index_d(src, v, N)] = cap[index_d(src, v, N)];
        flow[index_d(v, src, N)] = -flow[index_d(src, v, N)];
        excess[v] = flow[index_d(src, v, N)];
    }
}


__global__ void relabel_excess_d(int64_t *excess, int *stash_excess, int N){
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int nthread = blockDim.x * gridDim.x;
    for (int v = tid; v < N; v += nthread) {
        if (stash_excess[v] != 0) {
            excess[v] += stash_excess[v];
            stash_excess[v] = 0;
        }
    }
}


__global__ void relabel_d(int *dist, int64_t *excess, int *cap, int *flow, int N, int src, int *active_nodes, int count, int *stash_dist) {
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


__global__ void push_d(int *dist, int64_t *excess, int *cap, int *flow, int N, int src, int *active_nodes, int count, int *stash_excess) {
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

    int *dist = (int *) calloc(N, sizeof(int));
    int *dist_d, *stash_dist_d;
    cudaMalloc(&dist_d, N * sizeof(int));
    cudaMalloc(&stash_dist_d, N * sizeof(int));

    int64_t *excess = (int64_t *) calloc(N, sizeof(int64_t));
    int64_t *excess_d;
    cudaMalloc(&excess_d, N * sizeof(int64_t));

    int *stash_excess = (int *) calloc(N, sizeof(int));
    int *stash_excess_d;
    cudaMalloc(&stash_excess_d, N * sizeof(int));
    
    int *cap_d, *flow_d;
    cudaMalloc(&cap_d, N * N * sizeof(int));
    cudaMalloc(&flow_d, N * N * sizeof(int));
    
	cudaMemcpy(cap_d, cap, sizeof(int) * N * N, cudaMemcpyHostToDevice);
    pre_flow_d<<<blocks_per_grid, threads_per_block>>>(dist_d, excess_d, cap_d, flow_d, N, src);

    vector<int> active_nodes;
    int *active_nodes_d;
    cudaMalloc(&active_nodes_d, N * sizeof(int));

    int *stash_send = (int *) calloc(N * N, sizeof(int));
    for (auto u = 0; u < N; u++) {
        if (u != src && u != sink) {
            active_nodes.emplace_back(u);
            //std::cout << "u" << u << std::endl;
        }
    }
    
    // Four-Stage Pulses.
    while (!active_nodes.empty()) {
        int count = active_nodes.size();
        cudaMemcpy(active_nodes_d, active_nodes.data(), sizeof(int) * count, cudaMemcpyHostToDevice);
        push_d<<<blocks_per_grid, threads_per_block>>>(dist_d, excess_d, cap_d, flow_d, N, src, active_nodes_d, count, stash_excess_d);
        
		cudaMemcpy(stash_dist_d, dist_d, sizeof(int) * N, cudaMemcpyDeviceToDevice);
		relabel_d<<<blocks_per_grid, threads_per_block>>>(dist_d, excess_d, cap_d, flow_d, N, src, active_nodes_d, count, stash_dist_d);
		swap(dist_d, stash_dist_d);
        
		relabel_excess_d<<<blocks_per_grid, threads_per_block>>>(excess_d, stash_excess_d, N);
		cudaMemcpy(excess, excess_d, sizeof(int64_t) * N, cudaMemcpyDeviceToHost);
        cudaMemcpy(flow, flow_d, sizeof(int) * N * N, cudaMemcpyDeviceToHost);
        
		active_nodes.clear();
        for (auto u = 0; u < N; u++) {
            if (excess[u] > 0 && u != src && u != sink) {
                active_nodes.emplace_back(u);
            }
        }
    }

    free(dist);
    free(excess);
    free(stash_send);
    free(stash_excess);

    return 0;
}
