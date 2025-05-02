#!/bin/bash

## TRAINING
python3 -m pdb -c continue -m torch.distributed.launch \
    --nproc_per_node=1 \
    scripts/train_net.py \
    --config-file "experiments/vidstg.yaml" \
    INPUT.RESOLUTION 64 \
    OUTPUT_DIR output/vidstg \
    TENSORBOARD_DIR output/vidstg \
    | tee output.txt

## EVALUATION
# python3 -m pdb -c continue -m torch.distributed.launch \
#     --nproc_per_node=1 \
#     scripts/test_net.py \
#     --config-file "experiments/vidstg.yaml" \
#     INPUT.RESOLUTION 32 \
#     MODEL.WEIGHT "output/vidstg/model_final.pth" \
#     OUTPUT_DIR output/vidstg
