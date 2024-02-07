# Set common variables
dataset="c4"
sparsity_ratio=0.5
cuda_device=0

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

# Define function to run python command
run_compress () {
    python main.py \
    --model $1 \
    --dataset $dataset \
    --sparsity_ratio $sparsity_ratio \
    --sparsity_type "2:4" \
    --recover \
    --compress \
    --alpha $2 \
    --save_model $3
}

run_compress "facebook/opt-1.3b" "0.25" "./WRP/opt/opt-1.3b/"
run_compress "facebook/opt-2.7b" "0.25" "./WRP/opt/opt-2.7b/"
run_compress "facebook/opt-6.7b" "0.25" "./WRP/opt/opt-6.7b/"
run_compress "facebook/opt-13b" "0.25" "./WRP/opt/opt-13b/"
run_compress "facebook/opt-30b" "0.25" "./WRP/opt/opt-30b/"
