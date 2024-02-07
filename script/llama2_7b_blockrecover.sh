# Set common variables
model="meta-llama/Llama-2-7b-hf"
dataset="pajama"
sparsity_ratio=0.5
cuda_device=0

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

# Define function to run python command
run_with_blockrecover () {
    python main.py \
    --model $model \
    --dataset $dataset \
    --sparsity_ratio $sparsity_ratio \
    --sparsity_type "2:4" \
    --log $1 \
    --recover \
    --recover_format "block_ELL" \
    --blocksize $2 \
    --topn $3
}

run_with_blockrecover "llama2_7b_block.csv"  "64" "4"
run_with_blockrecover "llama2_7b_block.csv"  "64" "8"
run_with_blockrecover "llama2_7b_block.csv"  "32" "8"
run_with_blockrecover "llama2_7b_block.csv"  "32" "16"
run_with_blockrecover "llama2_7b_block.csv"  "16" "16"
run_with_blockrecover "llama2_7b_block.csv"  "16" "32"

