#!/bin/sh

python inference.py \
  -w "checkpoints/state_dict--epoch=30.ckpt" \
  -o "data/output" \
  -i "data/DICM/10.jpg" -s 128 -c 4 \
  --verbose
