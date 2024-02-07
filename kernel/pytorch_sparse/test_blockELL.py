import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def recover_top_n_blocks(W_metric, blocksize, n, W):
    #check blocksize
    assert W_metric.size(0) % blocksize == 0, "Rows must be divisible by blocksize"
    assert W_metric.size(1) % blocksize == 0, "Columns must be divisible by blocksize"

    #split W_metric: torch.tensor.unfold(dim, size, step)
    blocks = W_metric.unfold(0, blocksize, blocksize).unfold(1, blocksize, blocksize) 
    # '*': parameters unpack
    block_sums = blocks.contiguous().view(*blocks.shape[:2], -1).sum(-1)
    top_n_indices = block_sums.topk(n, dim=1).indices
    result = (torch.zeros_like(W_metric) == 1)
    #copy the blocks from W
    for i in range(top_n_indices.shape[0]):
        row = i * blocksize
        for j in range(top_n_indices.shape[1]):
            col = top_n_indices[i][j] * blocksize
            result[row:row + blocksize, col:col + blocksize] = W[row:row + blocksize, col:col + blocksize]
    top_n_indices, _ = torch.sort(top_n_indices, dim=1)
    return result, top_n_indices

W_row = 4096
W_col = W_row
seq_len = 2048
print("seq_len: %d, hidden_state: %d"%(seq_len, W_col))
blocksize = 64
topn = 8
print("sparsity: %.2f%%"%(topn/(W_col/blocksize)*100))
COUNT = 1000
device = torch.device("cuda")      
W = torch.rand(W_row, W_col).half().to(device)
x = torch.rand(seq_len, W_col).half().to(device)
result, index = recover_top_n_blocks(W, blocksize, topn, W)
W_dense = torch.where(result, W, 0.0)  
#####################################################################
################### Dense ########################################

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

res = torch.matmul(x, W_dense.t())

for _ in range(128):
    torch.matmul(x, W_dense.t(), out=res)

torch.cuda.synchronize()
start_event.record()
for _ in range(COUNT):
    torch.matmul(x, W_dense.t(), out=res)
    torch.cuda.synchronize()
end_event.record()
time_ms = start_event.elapsed_time(end_event) / COUNT
print("Dense-Dense time: %.3f ms"%time_ms)


#####################################################################
################### blockELL ########################################
hA_values = W[result]
hA_indices = index.reshape(1, -1).to(torch.int32)

from spmm_block_ell import blockELLSpmm_cuSparse, create_cuSparse_Buffer, delete_cuSparse_Buffer
create_cuSparse_Buffer(x, hA_values, hA_indices, W_row, W_col, blocksize, blocksize*topn, res)

for _ in range(128):
    blockELLSpmm_cuSparse(x, hA_values, hA_indices, W_row, W_col, blocksize, blocksize*topn, res)

torch.cuda.synchronize()
start_event.record()
for _ in range(COUNT):
    blockELLSpmm_cuSparse(x, hA_values, hA_indices, W_row, W_col, blocksize, blocksize*topn, res)
    torch.cuda.synchronize()
end_event.record()
time_ms = start_event.elapsed_time(end_event) / COUNT
print("My Block ELL time: %.3f ms"%time_ms)

delete_cuSparse_Buffer()
