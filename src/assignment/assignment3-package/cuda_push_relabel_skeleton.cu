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

__global__ void pre_flow(int *dist, unsigned long long int *excess, int *cap, int *flow, int N, int src) {
    int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    int element_skip = blockDim.x * gridDim.x;

    if (global_tid == 0 && blockIdx.x == 0) {
        dist[src] = N;
    }

    for (int v = global_tid; v < N; v += element_skip) {
        flow[utils::dev_idx(src, v, N)] = cap[utils::dev_idx(src, v, N)];
        flow[utils::dev_idx(v, src, N)] = -flow[utils::dev_idx(src, v, N)];
        excess[v] = flow[utils::dev_idx(src, v, N)];
    }
}


/*
 * NOTE: below there are two version of `push` function. Both are OK for us.
 * The second one is slight faster than the first one
 *
 */

__global__ void push(int* active_nodes, int active_nodes_size, int* dist, unsigned long long int *excess,
                     int* stash_send, int *cap, int *flow, int N) {

    for (int nodes_it = blockIdx.x; nodes_it < active_nodes_size; nodes_it += gridDim.x) {
        auto u = active_nodes[nodes_it];
        extern __shared__ int v_can_push_excess[];

        __shared__ int v_can_push_count;

        if (threadIdx.x == 0){
            v_can_push_count = 0;
        }
        __syncthreads();

        for (int v = threadIdx.x; v < N; v += blockDim.x) {
            auto residual_cap = cap[utils::dev_idx(u, v, N)] - flow[utils::dev_idx(u, v, N)];
            if (residual_cap > 0 && dist[u] > dist[v] && excess[u] != 0) {
                int this_pos = atomicAdd(&v_can_push_count, 2);
                v_can_push_excess[this_pos] = residual_cap;
                v_can_push_excess[this_pos + 1] = v;
            }
        }
        __syncthreads();

        if(threadIdx.x == 0){
            for (int v = 0; v < v_can_push_count && excess[u] != 0; v +=2){
                if(v_can_push_excess[v] > 0){
                    auto send = v_can_push_excess[v] < excess[u] ? v_can_push_excess[v]:excess[u];
                    auto new_excess = excess[u] - send;
                    excess[u] = new_excess;
                    stash_send[utils::dev_idx(u, v_can_push_excess[v+1], N)] = send;
                }
            }
        }
        __syncthreads();
    }
}

//__global__ void push(int* active_nodes, int active_nodes_size, int* dist, unsigned long long int *excess,
//                     int* stash_send, int *cap, int *flow, int N) {
//    for (int nodes_it = blockIdx.x; nodes_it < active_nodes_size; nodes_it += gridDim.x) {
//        auto u = active_nodes[nodes_it];

//        for (int v = threadIdx.x; v < N; v += blockDim.x) {
//            auto residual_cap = cap[utils::dev_idx(u, v, N)] - flow[utils::dev_idx(u, v, N)];

//            if (residual_cap > 0 && dist[u] > dist[v] && excess[u] != 0) {
//                unsigned long long int send;
//                unsigned long long int old_excess;
//                unsigned long long int new_excess;

//                do {
//                    old_excess = excess[u];
//                    send = old_excess < residual_cap ? old_excess : residual_cap;
//                    new_excess = old_excess - send;
//
//                    auto tmp = atomicCAS(excess + u, old_excess, new_excess);

//                    if (tmp == old_excess) {
//                        stash_send[utils::dev_idx(u, v, N)] = send;
//                        break;
//                    }

//                } while (excess[u] != 0);
//            }
//        }
//    }
//}

__global__ void apply_push_stash(int* active_nodes, int active_nodes_size, unsigned long long int *stash_excess,
                                 int* stash_send, int *flow, int N) {

    for (int nodes_it = blockIdx.x; nodes_it < active_nodes_size; nodes_it += gridDim.x) {
        auto u = active_nodes[nodes_it];

        for (int v = threadIdx.x; v < N; v += blockDim.x) {
            if (stash_send[utils::dev_idx(u, v, N)] > 0) {
                auto send = stash_send[utils::dev_idx(u, v, N)];
                flow[utils::dev_idx(u, v, N)] += send;
                flow[utils::dev_idx(v, u, N)] -= send;
                atomicAdd(stash_excess + v, send);
                stash_send[utils::dev_idx(u, v, N)] = 0;
            }
        }
    }
}

__global__ void relabel(int* active_nodes, int active_nodes_size, unsigned long long int *excess, int* dist,
                        int* dist_stash, int *cap, int *flow, int N) {

    for (int nodes_it = blockIdx.x; nodes_it < active_nodes_size; nodes_it += gridDim.x) {
        auto u = active_nodes[nodes_it];

        if (excess[u] != 0) {
            __shared__ int min_dist;

            if (threadIdx.x == 0) {
                min_dist = INT32_MAX;
            }
            __syncthreads();

            for (int v = threadIdx.x; v < N; v += blockDim.x) {
                auto residual_cap = cap[utils::dev_idx(u, v, N)] - flow[utils::dev_idx(u, v, N)];
                if (residual_cap > 0) {
                    atomicMin(&min_dist, dist[v]);
                }
            }
            __syncthreads();

            if(threadIdx.x == 0) {
                dist_stash[u] = min_dist + 1;
            }
        }
    }
}

__global__ void update_excess(unsigned long long int *excess, unsigned long long int *stash_excess, int N) {
    int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    int element_skip = blockDim.x * gridDim.x;

    for (int v = global_tid; v < N; v += element_skip) {
        if (stash_excess[v] != 0) {
            excess[v] += stash_excess[v];
            stash_excess[v] = 0;
        }
    }
}

int push_relabel(int blocks_per_grid, int threads_per_block, int N, int src, int sink, int *cap, int *flow) {
    dim3 blocks(blocks_per_grid);
    dim3 threads(threads_per_block);

    int* d_dist_even;
    int* d_dist_odd;
    uint64_t *excess = (uint64_t*) malloc(N * sizeof(uint64_t));

    unsigned long long int* d_excess;
    unsigned long long int* d_stash_excess;
    int* d_cap;
    int* d_flow;
    int* d_stash_send;
    GPUErrChk(cudaMalloc(&d_dist_even, sizeof(int) * N));
    GPUErrChk(cudaMalloc(&d_dist_odd, sizeof(int) * N));
    cudaMalloc(&d_excess, sizeof(long long int) * N);
    cudaMalloc(&d_stash_excess, sizeof(long long int) * N);

    cudaMalloc(&d_cap, sizeof(int) * N * N);
    cudaMalloc(&d_flow, sizeof(int) * N * N);
    cudaMalloc(&d_stash_send, sizeof(int) * N * N);

    cudaMemset(d_dist_even, 0, sizeof(int) * N);
    cudaMemset(d_stash_send, 0, sizeof(int) * N * N);
    cudaMemcpy(d_cap, cap, sizeof(int) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_flow, flow, sizeof(int) * N * N, cudaMemcpyHostToDevice);

    //GPUErrChk(cudaDeviceSynchronize());
    // PreFlow.
    pre_flow<<<blocks, threads>>>(d_dist_even, d_excess, d_cap, d_flow, N, src);

    int* d_active_nodes;
    cudaMalloc(&d_active_nodes, sizeof(int) * N);

    int* active_nodes = (int*) malloc(N * sizeof(int));
    int active_nodes_size = 0;

    for (auto u = 0; u < N; u++) {
        if (u != src && u != sink) {
            active_nodes[active_nodes_size++] = u;
        }
    }

    cudaMemcpy(d_active_nodes, active_nodes, sizeof(int) * N, cudaMemcpyHostToDevice);
    //GPUErrChk(cudaDeviceSynchronize());
    auto round = 0;
    auto iter = 0;
    // Four-Stage Pulses.
    while (active_nodes_size != 0) {
        // Stage 1: push.
        // Push Kernel.
        int*& d_dist = (round == 0) ? d_dist_even : d_dist_odd;
        int*& d_dist_stash = (round == 1) ? d_dist_even : d_dist_odd;
        push<<<blocks, threads, 2 * N * sizeof(int)>>>(d_active_nodes, active_nodes_size, d_dist, d_excess, d_stash_send, d_cap, d_flow, N);
        //GPUErrChk(cudaDeviceSynchronize());

        apply_push_stash<<<blocks, threads>>>(d_active_nodes, active_nodes_size, d_stash_excess, d_stash_send, d_flow, N);
        //GPUErrChk(cudaDeviceSynchronize());

        // Stage 2, 3: relabel (update dist to stash_dist).
        cudaMemcpy(d_dist_stash, d_dist, sizeof(int) * N, cudaMemcpyDeviceToDevice);
        // Relabel Kernel.
        relabel<<<blocks, threads>>>(d_active_nodes, active_nodes_size, d_excess, d_dist, d_dist_stash, d_cap, d_flow, N);
        //GPUErrChk(cudaDeviceSynchronize());

        // Stage 4: apply excess-flow changes for destination vertices.
        update_excess<<<blocks, threads>>>(d_excess, d_stash_excess, N);
        //GPUErrChk(cudaDeviceSynchronize());

        // Construct active nodes.
        cudaMemcpy(excess, d_excess, sizeof(uint64_t) * N, cudaMemcpyDeviceToHost);
        active_nodes_size = 0;

        for (auto u = 0; u < N; u++) {
            if (excess[u] != 0 && u != src && u != sink) {
                active_nodes[active_nodes_size++] = u;
            }
        }

        if (active_nodes_size > 0) {
            cudaMemcpy(d_active_nodes, active_nodes, sizeof(int) * N, cudaMemcpyHostToDevice);
        }

        round = 1 - round;

        iter++;
    }

    cudaMemcpy(flow, d_flow, sizeof(int) * N * N, cudaMemcpyDeviceToHost);

    free(excess);
    free(active_nodes);

    cudaFree(d_cap);
    cudaFree(d_flow);
    cudaFree(d_active_nodes);
    cudaFree(d_dist_odd);
    cudaFree(d_dist_even);
    cudaFree(d_excess);
    cudaFree(d_stash_excess);
    cudaFree(d_stash_send);

    return 0;
}