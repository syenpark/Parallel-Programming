# Parallel Programming

- Introduction to parallel computer architectures- Principles of parallel algorithm design- Shared-memory programming models- Message passing programming models- Data-parallel programming models for GPUs- Case studies of parallel algorithms, systems, and applications- Hands-on experience with writing parallel programs for tasks of interest

## Message Passing Programming Models
### Message Passing Interface (MPI)
## Shared Memory Programming Models
### Pthreads
### OpenMP
## Data Parallel Programming Models
### CUDA

All assignments are to make a serial push-relabel algorithm parallelised with MPI, Pthreads, CUDA.  

## Assignment DescriptionThe push–relabel algorithm (alternatively, preflow–push algorithm) is an algorithm for finding maximum flows from a single source vertex to a sink vertex in a directed weighted graph. The weight on each edge represents the capacity of the edge. The algorithm starts from the source and pushes a number (the flow value) that is no more than the edge weight to each neighbor. This process goes on until all vertices have no excess flows (the incoming flow is equal to the outgoing flow). The resulting flows in the graph are the maximum flows.  The Maximum Flow Problem. Given source and sink vertices (src, sink) and a capacity matrix C of a directed graph G=(V, E) where the in-degree of src is zero, and the out-degree of sink is zero; find the maximum flows F from the source vertex src to the sink vertex sink with two constraints.  • Flow on an edge doesn’t exceed the given capacity of the edge. (F[v][w] ≤ C[v][w]).• Incoming flow is equal to outgoing flow for every vertex except src and sink.The pseudo code of a push-relabel algorithm is given in Algorithm 1. The input parameters are the capacity matrix C, source and sink vertices src and sink. Four vertex properties are introduced for the push-relabel operations namely: (1) distance labels dist, (2) stashed distance labels stash_dist, (3) excess flows excess and (4) stashed excess flows stash_excess.  The algorithm starts with an initialization of the distance labels dist, excess flows excess and the flow matrix F (Lines 1-4). After the preflow operation, the algorithm conducts iterative computations on each active node whose excess flow is greater than zero. The set of active nodes is denoted by Q. In each iteration, there are four stages: (1)excess flow pushing, (2) distance relabeling, (3) stashed distance change applying and (4) stashed excess flow change applying.