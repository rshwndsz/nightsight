#!/usr/bin/env python3
import torch
from nightsight import model

# Load checkpoint
checkpoint = torch.load("checkpoints/state_dict--epoch=30.ckpt")

# An instance of your model.
net = model.EnhanceNetNoPool()

# Load weights
net.load_state_dict(checkpoint)

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 256, 256)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(net, example)

# Serialising script module to a file
traced_script_module.save("traced_EnhanceNetNoPool.pt")
