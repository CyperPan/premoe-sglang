// comm_utils.cpp - NCCL communication utilities for Pre-MoE
// Creates standalone NCCL communicators and reusable CUDA streams for async send/recv
#include <torch/extension.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

#define NCCL_CHECK(cmd) do {                          \
  ncclResult_t r = cmd;                               \
  if (r != ncclSuccess) {                             \
    throw std::runtime_error(                         \
      std::string("NCCL error: ") + ncclGetErrorString(r)); \
  }                                                   \
} while(0)

// Get a unique NCCL ID (call on rank 0, broadcast to others)
std::vector<uint8_t> get_nccl_unique_id() {
    ncclUniqueId id;
    NCCL_CHECK(ncclGetUniqueId(&id));
    return std::vector<uint8_t>(id.internal, id.internal + sizeof(id.internal));
}

// Create a new NCCL communicator from a unique ID
uintptr_t create_nccl_comm(std::vector<uint8_t> id_bytes, int rank, int world_size) {
    ncclUniqueId id;
    memcpy(id.internal, id_bytes.data(), sizeof(id.internal));
    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, world_size, id, rank));
    return reinterpret_cast<uintptr_t>(comm);
}

// Destroy communicator
void destroy_nccl_comm(uintptr_t comm_ptr) {
    ncclComm_t comm = reinterpret_cast<ncclComm_t>(comm_ptr);
    ncclCommDestroy(comm);
}

// Create a reusable CUDA stream for async communication
uintptr_t create_cuda_stream() {
    cudaStream_t stream;
    cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, -1);
    return reinterpret_cast<uintptr_t>(stream);
}

// Destroy a CUDA stream
void destroy_cuda_stream(uintptr_t stream_ptr) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    cudaStreamDestroy(stream);
}

// Start async send+recv on a pre-created CUDA stream
// The stream must be created with create_cuda_stream() and reused across calls
void async_send_recv_start(
    torch::Tensor send_data,
    torch::Tensor recv_data,
    uintptr_t comm_ptr,
    int peer_rank,
    uintptr_t stream_ptr)
{
    ncclComm_t comm = reinterpret_cast<ncclComm_t>(comm_ptr);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    size_t bytes = send_data.nbytes();

    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclSend(send_data.data_ptr(), bytes, ncclInt8, peer_rank, comm, stream));
    NCCL_CHECK(ncclRecv(recv_data.data_ptr(), bytes, ncclInt8, peer_rank, comm, stream));
    NCCL_CHECK(ncclGroupEnd());
}

// Wait for async communication to complete (does NOT destroy stream)
void async_send_recv_wait(uintptr_t stream_ptr) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    cudaStreamSynchronize(stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_nccl_unique_id", &get_nccl_unique_id, "Get NCCL unique ID");
    m.def("create_nccl_comm", &create_nccl_comm, "Create NCCL communicator");
    m.def("destroy_nccl_comm", &destroy_nccl_comm, "Destroy NCCL communicator");
    m.def("create_cuda_stream", &create_cuda_stream, "Create reusable CUDA stream");
    m.def("destroy_cuda_stream", &destroy_cuda_stream, "Destroy CUDA stream");
    m.def("async_send_recv_start", &async_send_recv_start, "Start async send/recv on pre-created stream");
    m.def("async_send_recv_wait", &async_send_recv_wait, "Wait for async send/recv");
}
