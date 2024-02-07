import torch
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
SparseSemiStructuredTensor._FORCE_CUTLASS = True

################ generate original data ##########################
COUNT=1000
bz = 1
seq_len = 2048
hidden_state = 4096
device = torch.device("cuda")         
x = torch.rand(bz, seq_len, hidden_state).half().to(device)
print("seq_len: %d, hidden_state: %d"%(seq_len, hidden_state))
W_semi_col = hidden_state
W_semi_row = hidden_state
W_semi = torch.Tensor([0, 0, 1.1, 1.5]).tile((W_semi_row, W_semi_col//4)).half().to(device)

res = torch.zeros(hidden_state, seq_len).half().to(device)
###############################################################################
################# covert to semi-structured data format ###############
W_semi_structured = to_sparse_semi_structured(W_semi)
################################################################################
################ test matmul x*W^T or W*x^T ####################################
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
for _ in range(128):
    res = torch.matmul(x, W_semi_structured.t())

torch.cuda.synchronize()
start_event.record()
for _ in range(COUNT):
    res = torch.matmul(x, W_semi_structured.t())
    torch.cuda.synchronize()
end_event.record()
time_ms = start_event.elapsed_time(end_event) / COUNT
print("Semi-structured time: %.3f ms"%time_ms)

#################################################################################
############### check y_semi and y_recover #####################################
for _ in range(128):
    torch.matmul(x, W_semi.t(), out=res)

torch.cuda.synchronize()
start_event.record()
for _ in range(COUNT):
    torch.matmul(x, W_semi.t(), out=res)
    torch.cuda.synchronize()
end_event.record()
time_ms = start_event.elapsed_time(end_event) / COUNT
print("Dense-Dense time: %.3f ms"%time_ms)

for _ in range(128):
    torch.matmul(x, W_semi.t(), out=res)

torch.cuda.synchronize()
start_event.record()
for _ in range(COUNT):
    torch.matmul(x, W_semi.t(), out=res)
    torch.cuda.synchronize()
end_event.record()
time_ms = start_event.elapsed_time(end_event) / COUNT
print("Dense-Dense time: %.3f ms"%time_ms)

