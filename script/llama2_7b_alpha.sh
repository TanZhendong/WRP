# Set common variables
model="meta-llama/Llama-2-7b-hf"
dataset="pajama"
sparsity_ratio=0.5
cuda_device=0

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

# Define function to run python command
run_with_recover () {
    python main.py \
    --model $model \
    --dataset $dataset \
    --sparsity_ratio $sparsity_ratio \
    --sparsity_type "2:4" \
    --log $1 \
    --recover \
    --alpha $2 
}

run_with_recover "llama2_7b_alpha.csv"  "0.1"
run_with_recover "llama2_7b_alpha.csv"  "0.15"
run_with_recover "llama2_7b_alpha.csv"  "0.2"
run_with_recover "llama2_7b_alpha.csv"  "0.25"
run_with_recover "llama2_7b_alpha.csv"  "0.3"
run_with_recover "llama2_7b_alpha.csv"  "0.35"
run_with_recover "llama2_7b_alpha.csv"  "0.4"