import argparse
import torch
import numpy as np
import cv2
from torchvision.utils import save_image
from pathlib import Path

from nightsight import model
from nightsight import data
from nightsight.log import Logger
logger = Logger()

IMAGE_SIZE = (256, 256)


def inference(args):
    # Load state dict from local disk
    checkpoint = torch.load(args.weights)
    # Get model
    net = model.EnhanceNetNoPool()
    # Load state dict into model
    net.load_state_dict(checkpoint)
    # Toggle eval mode to avoid gradient computation
    net.eval()

    for image_path in args.images:
        # TODO Split image into patches, compute enhaced parallely and average
        # Load
        image = cv2.imread(image_path)
        # Resize
        image = cv2.resize(image, IMAGE_SIZE)
        # Tensorify and convert to [C H W]
        image = torch.from_numpy(image).permute(2, 0, 1)
        # Normalize
        image = torch.div(image, torch.Tensor([255.0]))
        # Convert to [N C H W]
        image = image.unsqueeze(0)
        # Enhance
        ei_1, ei, A = net(image)
        ei = ei.detach()[0] # Convert from [1 C H W] to [C H W]
        # Unnormalize
        # TODO See when this is required/not
        # ei = torch.clamp(ei * torch.Tensor([255]), 0, 255)
        logger.debug(f"ei: {ei.min()} -> {ei.max()}")
        # Save
        save_image(ei, Path(args.outdir) / image_path.split('/')[-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", required=True)
    parser.add_argument("-i", "--images", nargs='+', required=True)
    parser.add_argument("-o", "--outdir", default="./data/output/")
    args = parser.parse_args()
    inference(args)
