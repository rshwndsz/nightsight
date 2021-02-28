import torch
import numpy as np
import torchvision as tv
from tqdm import tqdm
from pathlib import Path
from zipfile import ZipFile
import tarfile
import os
import requests


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def download_file(url, destination_dir='./', desc=None, force=False):
    """
    Download a file from any url using requests
    """
    # Convert path to pathlib object if not already
    destination_dir = Path(destination_dir)
    # Get filename from url
    fname = url.split('/')[-1]
    # Construct path to file in local machine
    local_filepath = Path(destination_dir) / fname

    if local_filepath.is_file() and not force:
        logger.info(
            "File(s) already downloaded. Use force=True to download again.")
        return local_filepath
    else:
        # Safely create nested directory - https://stackoverflow.com/a/273227
        destination_dir.mkdir(parents=True, exist_ok=True)

    if desc is None:
        desc = f"Downloading {fname}"

    # Download large file with requests - https://stackoverflow.com/a/16696317
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size_in_bytes = int(r.headers.get('content-length', 0))
        block_size = 1024
        # Progress bar for downloading file - https://stackoverflow.com/a/37573701
        pbar = tqdm(total=total_size_in_bytes,
                    unit='iB',
                    unit_scale=True,
                    desc=desc)
        with open(local_filepath, 'wb') as f:
            for data in r.iter_content(block_size):
                pbar.update(len(data))
                f.write(data)
        pbar.close()

    # TODO Add SHA256 or MD5 comparison

    return local_filepath


def extract_file(fname,
                 destination_dir="./",
                 ftype=None,
                 desc=None,
                 remove_extract=True):
    # Convert to pathlib objects
    fname = Path(fname)
    destination_dir = Path(destination_dir)

    # Check arguments
    if not fname.is_file():
        raise IOError(f"The file {str(fname)} does not exist.")

    # Safely create nested directory - https://stackoverflow.com/a/273227
    destination_dir.mkdir(parents=True, exist_ok=True)

    if desc is None:
        desc = f"Extracting {str(fname.name)}"

    # Get type of extract
    if ftype is None:
        # Takes care of '<name>.tar.gz'
        ftype = ''.join(fname.suffixes)

    # Extract the dataset into `destination_dir`
    if ftype == '.tar':
        # https://stackoverflow.com/a/30888321
        with tarfile.open(fname) as tar:
            tar.extractall(path=destination_dir)

    elif ftype == '.tgz' or ftype == '.tar.gz':
        # https://stackoverflow.com/a/30888321
        with tarfile.open(fname) as tar:
            tar.extractall(path=destination_dir)

    elif ftype == '.zip':
        # https://stackoverflow.com/a/56970565
        with ZipFile(fname, 'r') as zip:
            zip.extractall(path=destination_dir)

    else:
        raise IOError(f"The suffix: {ftype} is not supported.")

    # Delete the compressed dataset if requested (by default: yes)
    if remove_extract:
        os.remove(fname)


def make_grid(tensors, nrow=2, padding=2, isNormalized=True):
    """
    Convert a list of tensors into a numpy image grid
    """
    grid = tv.utils.make_grid(tensor=tensors.detach().cpu(),
                              nrow=nrow,
                              padding=padding,
                              normalize=(not isNormalized))
    if isNormalized:
        ndgrid = grid.mul(255).add_(0.5).clamp_(0, 255).permute(
            1, 2, 0).numpy().astype(np.uint16)
    else:
        ndgrid = grid.clamp_(0, 255).permute(1, 2, 0).numpy().astype(np.uint16)
    return ndgrid
