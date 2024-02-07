import torch
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
SparseSemiStructuredTensor._FORCE_CUTLASS = True

def generate_sparse_matrix(row, col, sparsity):
    matrix = torch.rand(row, col).half().to(device)  
    # threshold = 1.0 - sparsity
    mask = matrix > sparsity
    sparse_matrix = matrix * mask  
    density = torch.sum(sparse_matrix > 0).item() / (sparse_matrix.shape[0] * sparse_matrix.shape[1])
    print(f"Sparsity: {1 - density:.2f}")
    return sparse_matrix

################ generate original data ##########################
COUNT=1000
batch_size = 1
seq_len = 1
hidden_state = 4096
device = torch.device("cuda")         
x = torch.rand(batch_size, seq_len, hidden_state).half().to(device)
x = x.squeeze(0)

W_semi_col = hidden_state
W_semi_row = W_semi_col
sparsity = 0.99

W_recover = generate_sparse_matrix(W_semi_row, W_semi_col, sparsity)
###############################################################################
################# covert to coo and semi-structured data format ###############
W_recover_csr = W_recover.to_sparse_csr()
crow_indices_int32 = W_recover_csr.crow_indices().to(torch.int32)
col_indices_int32 = W_recover_csr.col_indices().to(torch.int32)
values = W_recover_csr.values()
W_recover_csr = torch.sparse_csr_tensor(crow_indices_int32, col_indices_int32, values, W_recover_csr.size(), dtype=values.dtype)
print("batch_size: %d, seq_len: %d, hidden_state: %d"%(batch_size, seq_len, hidden_state))
##############################################################################
################ test matmul x*W^T or W*x^T ####################################
for _ in range(128):
    y_recover = torch.matmul(x, W_recover_csr.t())


start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

torch.cuda.synchronize()
start_event.record()
for _ in range(COUNT):
    # y_recover = torch.matmul(W_recover_csr, x.t())
    y_recover = torch.matmul(x, W_recover_csr.t())
    torch.cuda.synchronize()
end_event.record()
time_ms = start_event.elapsed_time(end_event) / COUNT
print("Csr time: %.3f ms"%time_ms)

#################################################################################
############### check y_semi and y_recover #####################################
res = torch.zeros(batch_size, seq_len, hidden_state).half().to(device)
res = res.squeeze(0)
# res = torch.zeros(seq_len, hidden_state).half().to(device)
W_recover = torch.rand(W_semi_row, W_semi_col).half().to(device)  

for _ in range(128):
    torch.matmul(x, W_recover.t(), out=res)

torch.cuda.synchronize()
start_event.record()
for _ in range(COUNT):
    # torch.matmul(W_recover, x.t(), out=res)
    torch.matmul(x, W_recover.t(), out=res)
    torch.cuda.synchronize()
end_event.record()
time_ms = start_event.elapsed_time(end_event) / COUNT
print("Dense-Dense time: %.3f ms"%time_ms)
################################################################################
####################### test my kernel ##########################################
from spmm_csr import csrSpmm_cuSparse, create_cuSparse_Buffer, delete_cuSparse_Buffer

create_cuSparse_Buffer(x, W_recover_csr, res)
for _ in range(128):
    csrSpmm_cuSparse(x, W_recover_csr, res)
    # exit()

torch.cuda.synchronize()
start_event.record()
for _ in range(COUNT):
    csrSpmm_cuSparse(x, W_recover_csr, res)
    torch.cuda.synchronize()
end_event.record()
time_ms = start_event.elapsed_time(end_event) / COUNT
print("My Csr time: %.3f ms"%time_ms)
delete_cuSparse_Buffer()

