#!/bin/bash

# List of model sizes
models=("convnextv2_tiny.fcmae_ft_in1k" "convnextv2_small.fcmae_ft_in1k" \
        "convnextv2_base.fcmae_ft_in1k" "convnextv2_large.fcmae_ft_in1k")

# Loop through each model size
for model in "${models[@]}"; do
    echo "Training model: $model"
    python train_jersey_detector.py -m "$model"
done
