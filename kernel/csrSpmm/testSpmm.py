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
seq_len = 9
hidden_state = 8
device = torch.device("cuda")         
x = torch.rand(seq_len, hidden_state).half().to(device)
W_semi_col = hidden_state
W_semi_row = 4*W_semi_col
sparsity = 0.8

W_recover = generate_sparse_matrix(W_semi_row, W_semi_col, sparsity)
###############################################################################
############### create W_CSR #################################################
W_recover_csr = W_recover.to_sparse_csr()
crow_indices_int32 = W_recover_csr.crow_indices().to(torch.int32)
col_indices_int32 = W_recover_csr.col_indices().to(torch.int32)
values = W_recover_csr.values()
W_recover_csr = torch.sparse_csr_tensor(crow_indices_int32, col_indices_int32, values, W_recover_csr.size(), dtype=values.dtype)

######################################################################
##################test kernel ########################################
print(x.shape)

y = torch.matmul(x, W_recover.t())
print(y)
y = torch.matmul(x, W_recover_csr.t())
print(y)
res = torch.zeros_like(y)
print(W_recover_csr)


print("Start csrSpmm")
print(x.t().shape)
from spmm_csr import csrSpmm_cuSparse, create_cuSparse_Buffer, delete_cuSparse_Buffer
create_cuSparse_Buffer(x, W_recover_csr, res)
csrSpmm_cuSparse(x, W_recover_csr, res)
print(res)
delete_cuSparse_Buffer()