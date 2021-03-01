import torch
from nightsight import model


if __name__ == "__main__":
    # An instance of your model.
    net = model.EnhanceNetNoPool()

    # An example input you would normally provide to your model's forward() method.
    example = torch.rand(1, 3, 256, 256)

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(net, example)

    # Serialising script module to a file
    traced_script_module.save("traced_EnhanceNetNoPool.pt")

