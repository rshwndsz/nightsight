import os
import glob
import pytest

import numpy as np
import torch
from torch.utils import data as D
import albumentations as A
from albumentations.pytorch import ToTensorV2

from nightsight import data


# See https://stackoverflow.com/a/63425096
@pytest.fixture(scope="session")
def prepare_BSDS300(tmp_path_factory):
    urls = ["https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"]
    destination_dir = tmp_path_factory.getbasetemp()
    data.GenericImageDS.download(urls, destination_dir)

    root = os.path.join(destination_dir, "BSDS300", "images", "train")
    image_glob = "*.jpg"
    tf = A.Compose([
        A.Resize(256, 256, interpolation=4, p=1),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2()
    ])
    ds = data.GenericImageDS(root, image_glob, train=True, transform=tf)
    return [ root, image_glob, tf, ds ]


@pytest.mark.slow
class TestBSDS300:
    def test_image_collection(self, prepare_BSDS300):
        root, image_glob, _, ds = prepare_BSDS300
        print(root, image_glob)
        nimages = len(glob.glob(os.path.join(root, image_glob)))
        assert len(ds) == nimages

    def test_output_size(self, prepare_BSDS300):
        _, _, _, ds = prepare_BSDS300
        for image in ds:
            assert tuple(image.shape) == (3, 256, 256)

    def test_output_range(self, prepare_BSDS300):
        _, _, _, ds = prepare_BSDS300
        for image in ds:
            assert torch.le(torch.max(image), torch.tensor([1.0]))
            assert torch.ge(torch.min(image), torch.tensor([0.0]))
