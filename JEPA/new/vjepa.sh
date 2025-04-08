#!bin/bash

cd JEPA
python -m evals.main --fname configs/evals/new/vitl16_k400_16x8x3.yaml --devices cuda:0
