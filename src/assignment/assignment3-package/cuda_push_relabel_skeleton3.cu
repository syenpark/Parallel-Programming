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

__global__ void push(int *active_nodes, int *active_node_size, int N, int *cap, int *flow, int *dist, unsigned long long *excess, int *stash_send){
	int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	int numBlock = gridDim.x;
	int numThread = blockDim.x;
	int v, u;
	extern __shared__ unsigned long long residual_cap[];

	for (int i = blockId; i < active_node_size[0]; i+=numBlock){
		u = active_nodes[i];

		for (v = threadId; v < N; v+=numThread){
			residual_cap[v] = cap[idx(u, v, N)] -
							flow[idx(u, v, N)];
		}
		__syncthreads();
		if (threadId == 0){
			v = 0;
			while(excess[u]>0 && v < N){
				if (residual_cap[v] > 0 && dist[u] > dist[v]){
					stash_send[idx(u, v, N)] = min(excess[u], residual_cap[v]);
                    excess[u] -= stash_send[idx(u, v, N)];
				}
				v++;
			}
		}
		__syncthreads();
	}
}

__global__ void relabel (int *active_nodes, int *active_node_size, int N, int *cap, int *flow, int *dist, int *stash_dist, unsigned long long *excess){
	int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	int numBlock = gridDim.x;
	int numThread = blockDim.x;
	int u, v;
	int64_t residual_cap;
	__shared__ int min_dist;
	for (int i = blockId; i < active_node_size[0]; i+=numBlock){
		u = active_nodes[i];
		if (excess[u] > 0){
			min_dist = INT32_MAX;
			__syncthreads();
			for (v = threadId; v < N; v+=numThread){
				residual_cap = cap[idx(u, v, N)] - flow[idx(u, v, N)];
				if (residual_cap > 0) {
					atomicMin(&min_dist, dist[v]);
				}
			}
			__syncthreads();
			if (threadId == 0){
				stash_dist[u] = min_dist + 1;
			}
			__syncthreads();
		}

	}
}

int push_relabel(int blocks_per_grid, int threads_per_block, int N, int src, int sink, int *cap, int *flow) {
    /*
     *  Please fill in your codes here.
     *
     *  push relabel shouldn't run on cpu
     */
    return 0;
}



