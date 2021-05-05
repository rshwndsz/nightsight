import argparse
from pathlib import Path
import logging
import random, string

import numpy as np
from PIL import Image
import torch
from torchvision.utils import save_image

from nightsight import model
from nightsight.log import initLogger
logger = logging.getLogger(__name__)


def generateRandomString(length):
    """https://stackoverflow.com/a/2030081"""
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


def patchify(image, split_size, overlap):
    def start_points(size, split_size, overlap=0):
        points = [0]
        stride = int(split_size * (1-overlap))
        counter = 1
        while True:
            pt = stride * counter
            if pt + split_size >= size:
                points.append(size - split_size)
                break
            else:
                points.append(pt)
            counter += 1
        return points

    Y_points = start_points(image.shape[0], split_size, overlap)
    X_points = start_points(image.shape[1], split_size, overlap)

    # Split into patches
    splits = []
    for i in Y_points:
        for j in X_points:
            split = image[i:i+split_size, j:j+split_size]
            splits.append(split)

    # N split_size split_size 3
    splits = np.asarray(splits)

    return splits, X_points, Y_points


def reconstruct(splits, split_size, overlap, image_size, X_points, Y_points):
    logger.info(f"Reconstructing image of shape {image_size}")
    result = torch.zeros(image_size, dtype=torch.float)

    count = 0
    for i in Y_points:
        for j in X_points:
            result[i:i+split_size, j:j+split_size] = splits[count]
            count += 1
    return result


def _inference(net, inp, patch_size, csize):
    if type(inp) == str: 
        image_path = inp
        # Path to ndarray
        logger.info(f"Loading ndarray from {image_path}")
        try:
            image = np.load(image_path)
        except (ValueError, IOError):
            image = None
            logger.info(f"{image_path} could'nt be opened with numpy.")

        # Path to image
        logger.info(f"Loading image from {image_path}")
        try:
            image = np.asarray(Image.open(image_path))
        except (ValueError, IOError):
            image = None
            logger.info(f"{image_path} couldn't be opened with PIL.")

        # Unsuccessful reading using both PIL and numpy
        if image is None:
            logger.error("Invalid path. Couldn't open with numpy and PIL.")
            raise ValueError("Invalid path.")

    elif type(inp) == np.ndarray:
        image = inp
        image_path = generateRandomString(10) + '.jpg'
    else:
        logger.error(f"Invalid inputs provided.")
        raise ValueError("Invalid input.")

    # Split/Resize
    if image.shape[0] > patch_size * 2 or image.shape[1] > patch_size * 2:
        original_shape = image.shape
        # Split image into multiple patches
        logger.info(f"Splitting image of shape {image.shape}..")
        image, X_points, Y_points = patchify(image, patch_size, 1/8.0)
        logger.info(f"Split into {image.shape[0]} patches.")
    else:
        # Add a dummy dimension
        image = np.expand_dims(image, 0)

    # Tensorify and convert to [N C H W] from [N H W C]
    logger.info("Converting to tensor...")
    image = torch.from_numpy(image).permute(0, 3, 1, 2)
    # Normalize
    image = torch.div(image, torch.Tensor([255.0]))

    # Enhance
    logger.info("Enhancing image...")
    if image.size(0) > csize:
        # Get batch numbers
        cnos = [0]
        while cnos[-1] + csize < image.size(0):
            cnos.append(cnos[-1] + csize)
        cnos.append(image.size(0))

        # Cycle through batches of 8
        ei = []
        for i in range(len(cnos)-1):
            logger.info(f"Enhancing image[{cnos[i]}:{cnos[i+1]}]")
            _, _ei, _ = net(image[cnos[i]:cnos[i+1]])
            ei.append(_ei.detach())
        ei = torch.cat(ei, dim=0)

    else:
        _, ei, _ = net(image)
        ei = ei.detach()

    # Unnormalize
    # TODO See when this is required/not
    # ei = torch.clamp(ei * torch.Tensor([255]), 0, 255)
    # logger.info(f"ei: {ei.min()} -> {ei.max()}")

    if ei.size(0) == 1:
        result = ei.squeeze(0)
    else:
        result = reconstruct(ei.permute(0, 2, 3, 1), 
                             patch_size, 1/8.0, original_shape, 
                             X_points, Y_points).squeeze(0).permute(2, 0, 1)

    return result, image_path


def inference(weights, images, patch_size, cycle_size, outdir):
    if type(images) != list:
        images = [images]

    logger.info("Loading weights...")
    # Load state dict from local disk
    checkpoint = torch.load(weights)
    # Get model
    net = model.EnhanceNetNoPool()
    # Load state dict into model
    net.load_state_dict(checkpoint)
    # Toggle eval mode to avoid gradient computation
    net.eval()

    results = []
    for inp in images:
        results.append(_inference(net, inp, patch_size, cycle_size))

    if outdir is not None:
        for (ei, image_path) in results:
            logger.info(f"Saving image {ei.shape}")
            # Save image into `outdir`
            Path(outdir).mkdir(parents=True, exist_ok=True)
            output_name = Path(outdir) / (image_path.split('/')[-1])
            save_image(ei, output_name)
            logger.info(f"Saved {output_name}")
    return results

def inference_GUI(weights, image, patch_size, cycle_size, outfile):
    #if type(images) != list:
    #    images = [images]

    logger.info("Loading weights...")
    # Load state dict from local disk
    checkpoint = torch.load(weights)
    # Get model
    net = model.EnhanceNetNoPool()
    # Load state dict into model
    net.load_state_dict(checkpoint)
    # Toggle eval mode to avoid gradient computation
    net.eval()

    #results = []
    #for inp in images:
    #    results.append(_inference(net, inp, patch_size, cycle_size))
    result = _inference(net, image, patch_size, cycle_size)[0]

    if outfile is not None:
        logger.info(f"Saving image {result.shape}")
        # Save image into `outdir`
        #Path(outdir).mkdir(parents=True, exist_ok=True)
        #output_name = Path(outdir) / (image_path.split('/')[-1])
        save_image(result, outfile)
        logger.info(f"Saved {outfile}")
        return True
    else:
        return result


if __name__ == "__main__":
    initLogger()
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", required=True)
    parser.add_argument("-s", "--size", type=int, default=128, help="Patch size") 
    parser.add_argument("-c", "--cycle", type=int, default=4, help="Cycle size")
    parser.add_argument("-i", "--images", nargs='+', required=True)
    parser.add_argument("-o", "--outdir", default="./data/output/")
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true")
    parser.set_defaults(verbose=False)
    args = parser.parse_args()
    if not args.verbose:
        logger.setLevel(logging.INFO)

    res = inference(args.weights, args.images, args.size, args.cycle, args.outdir)
    del res
