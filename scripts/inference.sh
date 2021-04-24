#!/bin/sh

python inference.py \
  -w "checkpoints/state_dict--epoch=30.ckpt" \
  -o "data/output" \
  -i "data/rpi_images/03032021-210335.jpg" -s 256 \
  --verbose
