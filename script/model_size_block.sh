# Set common variables
dataset="c4"
sparsity_ratio=0.5
cuda_device=1

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
    --recover_format "block_ELL" \
    --save_model $2 \
    --blocksize $3 \
    --topn $4 \
    --log "result.csv"
}

run_compress "facebook/opt-1.3b" "./WRP/block625/opt-1.3b/" "32" "4"
run_compress "facebook/opt-2.7b" "./WRP/block625/opt-2.7b/" "32" "5"
run_compress "facebook/opt-6.7b" "./WRP/block625/opt-6.7b/" "32" "8"
run_compress "facebook/opt-13b" "./WRP/block625/opt-13b/" "32" "10"
run_compress "facebook/opt-30b" "./WRP/block625/opt-30b/" "32" "14"

run_compress "facebook/opt-1.3b" "./WRP/block125/opt-1.3b/" "32" "8"
run_compress "facebook/opt-2.7b" "./WRP/block125/opt-2.7b/" "32" "10"
run_compress "facebook/opt-6.7b" "./WRP/block125/opt-6.7b/" "32" "16"
run_compress "facebook/opt-13b" "./WRP/block125/opt-13b/" "32" "20"
run_compress "facebook/opt-30b" "./WRP/block125/opt-30b/" "32" "28"
