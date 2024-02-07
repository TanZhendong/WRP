import numpy as np
import torch
import argparse
import pandas as pd
from lib.modelutils import get_model
from lib.datautils import get_loaders
from lib.prune import fake_prune, prune_compress
from lib.eval import perplexity_eval
import os 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--dataset', type=str, help='Calibration dataset', choices=['c4', 'pajama'])
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4", "8:16", "16:32", "32:64", "64:128", "128:256"]) #change
    parser.add_argument('--log', type=str, default=None, help='Log file name.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--compress', default=False, action="store_true", help='Whether compress the model.')
    parser.add_argument("--recover", default=False, action="store_true", help='Weight recover.')
    parser.add_argument("--recover_format", type=str, default="CSR", choices=["block_ELL", "CSR"], help='Which data format for weight recover sparse matrix.') #change
    parser.add_argument("--alpha", type=float, default=0, help='Recover ratio.')
    parser.add_argument("--blocksize", type=int, default=64, help='Block Size.')
    parser.add_argument("--topn", type=int, default=8, help='Topn Block to recover.')
    args = parser.parse_args()
    # print(args)
    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1] 
    print(f"loading llm model {args.model}")
    model = get_model(args.model)
    model.eval()
    # print(model)
    print("loading calibdation data")
    dataloader = get_loaders(args.dataset, nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,model_path=args.model)
    print("dataset loading complete")

    if args.recover_format == "CSR":
        data = {
            'Model': [],
            'PPL(Wikitext2)': [],
            'PPL(ptb)': [],
            'PPL(c4)': [],
            'Alpha': [],
            'Recover_sparsity': []
        }
    elif args.recover_format == "block_ELL":
        data = {
            'Model': [],
            'PPL(Wikitext2)': [],
            'PPL(ptb)': [],
            'PPL(c4)': [],
            'Block_size': [],
            'Topn': [],
            'Recover_sparsity': []
        }
    df = pd.DataFrame(data)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not args.compress:
        #fake prune
        recover_sparsity = fake_prune(args, model, dataloader, dev, prune_n, prune_m)
        ppls = {}
        datasets = ["wikitext2", "ptb", "c4"]
        if "ptb" in datasets and "llama" in args.model.lower():
            print("As the Transformers version updates, the PTB dataset might produce unstable evaluation results on the LLAMA model. The cause of this instability is currently unknown. For example:\
                  \n - https://github.com/huggingface/transformers/issues/27382 \
                  \n - https://github.com/Vahe1994/SpQR/issues/16 \
                  \n Therefore, we do not recommend using the PTB dataset to evaluate the results of the LLAMA model.")
        for dataset in datasets:
            testloader = get_loaders(
                dataset,
                model_path=args.model,
                seqlen=model.seqlen,
                eval_mode=True,
            )
            ppl = perplexity_eval(model, testloader, dev)
            ppls[dataset] = ppl
        if args.recover_format == "CSR":
            item = {
                'Model': model_name,
                'PPL(Wikitext2)': ppls['wikitext2'],
                'PPL(ptb)': ppls['ptb'],
                'PPL(c4)': ppls['c4'],
                'Alpha': args.alpha,
                'Recover_sparsity': recover_sparsity
            }
        elif args.recover_format == "block_ELL":
            item = {
                'Model': model_name,
                'PPL(Wikitext2)': ppls['wikitext2'],
                'PPL(ptb)': ppls['ptb'],
                'PPL(c4)': ppls['c4'],
                'Block_size': args.blocksize,
                'Topn': args.topn,
                'Recover_sparsity': recover_sparsity
            }
        df = df._append(item, ignore_index=True) 
        print(df)
        csv_file_path = './logs/' + args.log
        df.to_csv(csv_file_path, mode='a', index=False)
        if args.save_model != None:
            model.save_pretrained(args.save_model)
    else:
        #true compress
        prune_compress(args, model, dataloader, dev, prune_n, prune_m)
        print(model)
        if args.save_model != None:
            # model.save_pretrained(args.save_model)
            os.makedirs(args.save_model)
            tar = args.save_model + model_name + '_wrp.pth'
            # print(tar)
            # torch.save(model.state_dict(), './' + args.save_model + model_name + '_wrp.pth')
            torch.save(model, tar)

if __name__ == '__main__':
    main()