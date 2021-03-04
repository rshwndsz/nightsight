import argparse
from pathlib import Path
import logging

import numpy as np
from PIL import Image
import torch
from torchvision.utils import save_image

from nightsight import model
from nightsight.log import Logger
logger = Logger()


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
        logger.debug(f"Loading {image_path}")
        image = Image.open(image_path)
        # Resize
        image = image.resize((args.size, args.size))
        # Tensorify and convert to [C H W]
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1)
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
        Path(args.outdir).mkdir(parents=True, exist_ok=True)
        output_name = Path(args.outdir) / (image_path.split('/')[-1])
        logger.debug(f"Saving {output_name}")
        save_image(ei, output_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", required=True)
    parser.add_argument("-s", "--size", type=int, default=256)
    parser.add_argument("-i", "--images", nargs='+', required=True)
    parser.add_argument("-o", "--outdir", default="./data/output/")
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true")
    parser.set_defaults(verbose=False)
    args = parser.parse_args()
    if not args.verbose:
        logger.setLevel(logging.INFO)
    inference(args)
