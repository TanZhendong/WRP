import torch
import torch.nn as nn
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
import warnings
from spmm_block_ell import blockELLSpmm_cuSparse, create_cuSparse_Buffer, delete_cuSparse_Buffer
warnings.filterwarnings("ignore", category=UserWarning)
SparseSemiStructuredTensor._FORCE_CUTLASS = True


def pad_to_multiple_of_eight_3D(x):
    batch_size, rows, cols = x.shape
    pad_size = -rows % 8  # the negative sign gives us the remainder needed to reach the next multiple of 8
    if pad_size > 0:
        padding = torch.zeros(batch_size, pad_size, cols, dtype=x.dtype, device=x.device)  # match the device and the data type of x
        x_padded = torch.cat([x, padding], dim=1)
    else:
        x_padded = x
    return x_padded, pad_size

def pad_to_multiple_of_eight_2D(x):
    rows, cols = x.shape
    pad_size = -rows % 8  # the negative sign gives us the remainder needed to reach the next multiple of 8
    if pad_size > 0:
        padding = torch.zeros(pad_size, cols, dtype=x.dtype, device=x.device)  # match the device and the data type of x
        x_padded = torch.cat([x, padding], dim=0)
    else:
        x_padded = x
    return x_padded, pad_size

class WRPLinear(nn.Module): 
    def __init__(self, W_semi, W_recover, bias=None):
        super(WRPLinear, self).__init__()
        self.in_features = W_semi.shape[1]
        self.out_features = W_semi.shape[0]
        W_recover = W_recover.to_sparse_csr()
        crow_indices_int32 = W_recover.crow_indices().to(torch.int32)
        col_indices_int32 = W_recover.col_indices().to(torch.int32)
        values = W_recover.values()
        W_recover = torch.sparse_csr_tensor(crow_indices_int32, col_indices_int32, values, W_recover.size(), dtype=values.dtype)
        W_semi = to_sparse_semi_structured(W_semi)
        self.register_buffer('weight_recover',W_recover) 
        self.register_buffer('weight',W_semi) 

        if bias is not None:
            self.register_buffer('bias', bias) 
        else:
            self.register_buffer('bias', None) 

    def __repr__(self):
        # print layer information
        return f'WRPLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})'

    def forward(self, x):
        y_recover = torch.matmul(x, self.weight_recover.t())
        if len(x.shape) == 3:
            x_padded, pad_size = pad_to_multiple_of_eight_3D(x)
        elif len(x.shape) == 2:
            x_padded, pad_size = pad_to_multiple_of_eight_2D(x)
        y_semi = torch.matmul(x_padded, self.weight.t())
        # Truncate the padded rows from the final result to restore original size if necessary
        if pad_size > 0:
            if len(x.shape) == 3:
                y_semi = y_semi[:, :-pad_size]
            elif len(x.shape) == 2:
                y_semi = y_semi[:-pad_size]
        if self.bias == None:
            y = y_semi + y_recover
        else:
            y = y_semi + y_recover + self.bias
        return y

class WRPBlockLinear(nn.Module):
    def __init__(self, W_semi, recover_values, recover_indices, blocksize, topn, bias=None):
        super(WRPBlockLinear, self).__init__()
        self.in_features = W_semi.shape[1]
        self.out_features = W_semi.shape[0]
        self.blocksize = blocksize
        self.topn = topn
        self.register_buffer('recover_values',recover_values) 
        self.register_buffer('recover_indices',recover_indices) 
        W_semi = to_sparse_semi_structured(W_semi)
        self.register_buffer('weight',W_semi) 
        if bias is not None:
            self.register_buffer('bias', bias) 
        else:
            self.register_buffer('bias', None) 

    def __repr__(self):
        # print layer information
        return f'WRPBlockLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})'

    def forward(self, x):
        #Only support batch size = 1
        flag = False
        if len(x.shape) == 3:
            # x_padded, pad_size = pad_to_multiple_of_eight_3D(x)
            x = x.squeeze(0)
            x_padded, pad_size = pad_to_multiple_of_eight_2D(x)
            flag = True
        elif len(x.shape) == 2:
            x_padded, pad_size = pad_to_multiple_of_eight_2D(x)
        y_semi = torch.matmul(x_padded, self.weight.t())
        # Truncate the padded rows from the final result to restore original size if necessary
        if pad_size > 0:
            if len(x.shape) == 3:
                y_semi = y_semi[:, :-pad_size]
            elif len(x.shape) == 2:
                y_semi = y_semi[:-pad_size]
        y_recover = torch.zeros_like(y_semi)
        create_cuSparse_Buffer(x, self.recover_values, self.recover_indices, self.out_features, self.in_features, self.blocksize, self.blocksize*self.topn, y_recover)
        blockELLSpmm_cuSparse(x, self.recover_values, self.recover_indices, self.out_features, self.in_features, self.blocksize, self.blocksize*self.topn, y_recover)
        delete_cuSparse_Buffer()
        if flag:
            y_semi = y_semi.unsqueeze(0)
            y_recover = y_recover.unsqueeze(0)
        if self.bias == None:
            y = y_semi + y_recover
        else:
            y = y_semi + y_recover + self.bias
        return y

#TODO: FusedWRPLinear