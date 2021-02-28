import glob
import os
from pathlib import Path

from PIL import Image
import numpy as np
from torch.utils import data as D
import albumentations as A
from albumentations.pytorch import ToTensorV2

from nightsight import utils
from nightsight.log import Logger
logger = Logger()


class GenericImageDS(D.Dataset):
    def __init__(self,
                 root,
                 image_glob="*.jpg",
                 train=True,
                 transform=None,
                 min_image_dim=256):
        self.root = root
        self.image_glob = image_glob
        self.train = train
        self.min_image_dim = min_image_dim

        image_regex = os.path.join(self.root, self.image_glob)
        self.image_paths = glob.glob(image_regex)
        if not len(self.image_paths):
            raise ValueError(f"No image found using {image_regex}")

        self.transform = transform
        if self.transform is None:
            # Default set of transforms if none are provided
            self.transform = A.Compose([
                A.Resize(self.min_image_dim, self.min_image_dim, 4, True, 1),
                A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), p=1),
                ToTensorV2()
            ])
        logger.info(f"Total samples: {len(self.image_paths)}")

    @staticmethod
    def download(urls, destination_dir, force=False):
        destination_dir = Path(destination_dir)

        # Check validity of arguments
        if not destination_dir.is_dir():
            raise ValueError("Provide destination_dir")

        if urls is None:
            raise ValueError("Provide URL(s)")

        # Download & Extract
        for url in urls:
            fname = utils.download_file(url, destination_dir)
            utils.extract_file(fname, destination_dir)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = np.asarray(Image.open(image_path))
        image = self.transform(image=image)["image"]
        return image

    def __len__(self):
        return len(self.image_paths)


class ZeroDceDS(GenericImageDS):
    pass


class SIDD(GenericImageDS):
    """
    Urls from https://www.eecs.yorku.ca/~kamel/sidd/files/SIDD_URLs_Mirror_2.txt
    """
    pass


class DarmstadtNoise(GenericImageDS):
    pass


class LOL(GenericImageDS):
    pass


class ExDark(GenericImageDS):
    pass


class VV(GenericImageDS):
    pass


class VIPLowNoise(GenericImageDS):
    pass


class DICM(GenericImageDS):
    pass


class LIME(GenericImageDS):
    pass


class GoogleOpenImagesDS(GenericImageDS):
    pass


class BSDS300(GenericImageDS):
    """
    Reference: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
    URL: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz
    """
    pass


class BSDS500(GenericImageDS):
    """
    Reference: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html
    URL: http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz
    """
    pass