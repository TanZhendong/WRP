import time 
import heapq 
import torch 
import numpy as np
import torch.nn as nn
from .modelutils import * 
from .layerwrapper import WrappedGPT
from .wrpLinear import WRPLinear, WRPBlockLinear
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor

@torch.no_grad()
def get_inps(model, data_iterable, dev, nsamples=None):
    """mocks model launch to collect inputs to the first model layer"""
    # print("catching inputs from data", flush=True)

    layers = get_layers(model)

    nsamples = nsamples or 128 #parameters

    if isinstance(data_iterable, torch.Tensor):

        def batch_generator(testenc, seqlen, nsamples):
            for i in range(nsamples):
                batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(dev)
                yield batch

        data_iterable = batch_generator(data_iterable, model.seqlen, nsamples)

    emb = model.get_input_embeddings()
    emb_dev = emb.weight.device
    if emb_dev.type != "cuda":
        emb = emb.to(dev)
        # opt has other embeddings
        if model.config.model_type == "opt":
            model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
            if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
                model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    dev = emb.weight.device  # now default device is the one where the embeddings are.
    layer_dev = next(layers[0].parameters()).device
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)

    forward_arg_names = [
        "attention_mask",
    ]
    if model.config.model_type.lower() in FALCON_TYPES:
        forward_arg_names.append("alibi")

    cache = {"i": 0, "attention_mask": None, "alibi": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            for forward_arg_name in forward_arg_names:
                cache[forward_arg_name] = kwargs.get(forward_arg_name)
            raise ValueError

    layers[0] = Catcher(layers[0])
    saved_num_threads = torch.get_num_threads()
    torch.set_num_threads(min(16, saved_num_threads))
    for batch in data_iterable:
        try:
            if isinstance(batch, (list, tuple)):
                model(batch[0].to(dev))
            elif isinstance(batch, torch.Tensor):
                model(batch.to(dev))
        except ValueError:
            pass
    torch.set_num_threads(saved_num_threads)
    layers[0] = layers[0].module

    layers[0] = layers[0].to(layer_dev)
    model.get_input_embeddings().to(emb_dev)
    if model.config.model_type == "opt":
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(emb_dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(emb_dev)
    torch.cuda.empty_cache()

    forward_args = {k: cache[k] for k in forward_arg_names}
    return inps, forward_args

def check_sparsity(model, device=torch.device("cuda:0")):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    print('#'*50)
    layers = get_layers(model)
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i].to(device)
        subset = find_sublayers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()
        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")
        layer.cpu()
    model.config.use_cache = use_cache 
    print('#'*50)
    return float(count)/total_params 

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

def fake_prune(args, model, dataloader, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    #use wanda metric
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    with torch.no_grad():
        inps, forward_args = get_inps(model, dataloader, device)
        outs = torch.zeros_like(inps)
    #move to gpus
    inps, outs = inps.to(device), outs.to(device)
    for k, v in forward_args.items():
        forward_args[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    recover_sparsity = []
    ############## start prune #############################
    layers = get_layers(model)
    for i in range(len(layers)):
        layer = layers[i].to(device)
        subset = find_sublayers(layer)
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp
        #forward
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), **forward_args)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            X_norm2 = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            W_metric = torch.abs(subset[name].weight.data) * X_norm2
            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True) #sort every rows
                # unstructured pruning
                indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)] #sort_res[1] is index
                W_mask.scatter_(1, indices, True)

            if args.recover:
                ######################### CSR Sparse ########################################
                if args.recover_format == "CSR":
                    alpha_mask = (torch.zeros_like(W_metric) == 1)
                    alpha = args.alpha
                    sort_res = torch.sort(W_metric, dim=-1, descending=True, stable=True) 
                    indices = sort_res[1][:,:int(W_metric.shape[1]*alpha)] 
                    alpha_mask.scatter_(1, indices, True)
                    alpha_mask = torch.logical_and(alpha_mask, W_mask)
                    #alpha mask是要救活的元素
                    true_ratio = torch.sum(alpha_mask).item() / alpha_mask.numel()
                    print("recover ratio:", true_ratio)
                    recover_sparsity.append(true_ratio)
                    W_mask = torch.logical_and(W_mask, torch.logical_not(alpha_mask))
                ##############################################################################
                ##################### Block-ELL Sparse #######################################
                elif args.recover_format == "block_ELL":
                    W_metric[torch.logical_not(W_mask)] = 0 #Preserve weights metrics that have not been pruned
                    recover_mask, _ = recover_top_n_blocks(W_metric, args.blocksize, args.topn, W_mask)
                    W_mask[recover_mask] = False

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), **forward_args)[0]
        inps, outs = outs, inps
        #offload 
        layer.cpu()

    model.config.use_cache = use_cache 
    check_sparsity(model)
    torch.cuda.empty_cache()
    if args.recover:
        if args.recover_format == 'CSR':
            avg_recover_sparsity = sum(recover_sparsity) / len(recover_sparsity)
            return avg_recover_sparsity
        elif args.recover_format == "block_ELL":
            avg_recover_sparsity = args.topn / (model.config.hidden_size / args.blocksize)
            return avg_recover_sparsity
    else:
        return 0

def prune_compress(args, model, dataloader, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    #use wanda metric
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    with torch.no_grad():
        inps, forward_args = get_inps(model, dataloader, device)
        outs = torch.zeros_like(inps)
    #move to gpus
    inps, outs = inps.to(device), outs.to(device)
    for k, v in forward_args.items():
        forward_args[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    ############## start prune #############################
    layers = get_layers(model)
    for i in range(len(layers)):
        layer = layers[i].to(device)
        subset = find_sublayers(layer)
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp
        #forward
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), **forward_args)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            X_norm2 = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            W_metric = torch.abs(subset[name].weight.data) * X_norm2
            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True) 
                # unstructured pruning
                indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)] 
                W_mask.scatter_(1, indices, True)

            if args.recover:
                if args.recover_format == "CSR":
                ######################### CSR Sparse ########################################
                    alpha_mask = (torch.zeros_like(W_metric) == 1)
                    alpha = args.alpha
                    sort_res = torch.sort(W_metric, dim=-1, descending=True, stable=True) 
                    indices = sort_res[1][:,:int(W_metric.shape[1]*alpha)] 
                    alpha_mask.scatter_(1, indices, True)
                    alpha_mask = torch.logical_and(alpha_mask, W_mask)
                    true_ratio = torch.sum(alpha_mask).item() / alpha_mask.numel()
                    print("recover ratio:", true_ratio)
                    W_recover = subset[name].weight.data.clone()
                    alpha_mask = torch.logical_not(alpha_mask)
                    W_recover[alpha_mask] = 0
                elif args.recover_format == "block_ELL":
                ############################ Blocked-ELL #########################################
                    W_metric[torch.logical_not(W_mask)] = 0 #Preserve weights metrics that have not been pruned
                    tmp = (torch.zeros_like(W_metric) == 0)
                    recover_mask, indices = recover_top_n_blocks(W_metric, args.blocksize, args.topn, tmp)
                    recover_values = subset[name].weight.data.clone()
                    recover_values[torch.logical_not(W_mask)] = 0 #this would be in 2:4 pattern
                    recover_values = recover_values[recover_mask].reshape(-1, args.blocksize*args.topn)
                    recover_indices = indices.reshape(1, -1).to(torch.int32)


            W_semi = subset[name].weight.data.clone()
            W_semi[W_mask] = 0

            if args.recover:
                if args.recover_format == "CSR":
                    #change Linear to WRPLinear
                    if subset[name].bias != None:
                        wrplinear = WRPLinear(W_semi, W_recover, bias=subset[name].bias.data.clone())
                    else:
                        wrplinear = WRPLinear(W_semi, W_recover, bias=None)
                        
                    wrplinear.to(device)
                    if '.' in name:
                        parent_name, child_name = name.rsplit('.', 1)
                        parent_module = getattr(layer, parent_name)
                        setattr(parent_module, child_name, wrplinear)
                    else:
                        setattr(layer, name, wrplinear)
                elif args.recover_format == "block_ELL":
                    #change Linear to WRPBlockLinear
                    if subset[name].bias != None:
                        wrplinear = WRPBlockLinear(W_semi, recover_values, recover_indices, args.blocksize, args.topn, bias=subset[name].bias.data.clone())
                    else:
                        wrplinear = WRPBlockLinear(W_semi, recover_values, recover_indices, args.blocksize, args.topn, bias=None)
                        
                    wrplinear.to(device)
                    if '.' in name:
                        parent_name, child_name = name.rsplit('.', 1)
                        parent_module = getattr(layer, parent_name)
                        setattr(parent_module, child_name, wrplinear)
                    else:
                        setattr(layer, name, wrplinear)

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), **forward_args)[0]
        
        # layer.cpu() #offload
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
