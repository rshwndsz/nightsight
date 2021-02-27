import glob
import os
from pathlib import Path

from PIL import Image
import numpy as np
from torch.utils import data as D
import albumentations as A
from albumentations.pytorch import ToTensorV2

from nightsight.log import logger
from nightsight import utils


class ZeroDceDS(D.Dataset):
    def __init__(self,
                 root="data/train_data/",
                 image_glob="*.jpg",
                 train=True,
                 transform=None):
        self.root = root
        self.image_glob = image_glob
        self.train = train

        self.image_paths = glob.glob(os.path.join(self.root, self.image_glob))
        if not len(self.image_paths):
            raise ValueError(
                f"No images found in {os.path.join(self.root, self.image_glob)}"
            )

        self.transform = transform
        logger.info(f"Total samples: {len(self.image_paths)}")

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = np.asarray(Image.open(image_path))

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image

    def __len__(self):
        return len(self.image_paths)


class GoogleOpenImagesDS(D.Dataset):
    def __init__(self,
                 root,
                 regex="*.jpg",
                 train=True,
                 transform=None,
                 min_image_dim=256):
        self.root = Path(root)
        self.train = train
        self.transform = transform
        self.regex = regex
        self.sigma = sigma
        self.min_image_dim = min_image_dim

        # The entire Open-Images dataset is divided into multiple folders
        # So, collect the paths to those folders
        self.image_dirs = [
            os.path.join(root, name) for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))
        ]
        # Now, get the list of all images by reading all images in each image folder
        self.image_paths = []
        for image_dir in self.image_dirs:
            self.image_paths.extend(glob.glob(os.path.join(image_dir, regex)))

        if self.transform is None:
            # Default set of transforms if none are provided
            self.transform = A.Compose([
                A.Resize(self.min_image_dim, self.min_image_dim, 4, True, 1),
                A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), p=1),
                ToTensorV2()
            ])

    @staticmethod
    def download(urls, destination_dir="./data/open-images", force=False):
        destination_dir = Path(destination_dir)

        # Basic check to see if already downloaded
        if destination_dir.is_dir() and not force:
            logger.info(
                "Destination directory exists. Use force=True to force download."
            )
            return

        # Check validity of arguments
        if urls is None:
            raise ValueError("Provide URL(s)")

        # Download & Extract
        for url in urls:
            fname = utils.download_file(url, destination_dir)
            utils.extract_file(fname, destination_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Read path to image
        image_path = self.image_paths[index]
        # Read image
        image = Image.open(image_path)
        # Now you are assured that mininum image shape will be (self.min_image_dim, self.min_image_dim)
        if image.size[0] < self.min_image_dim or image.size[
                1] < self.min_image_dim:
            image = image.resize((self.min_image_dim, self.min_image_dim),
                                 Image.LANCZOS)
        # Data augmentation
        image = self.transform(image=np.asarray(image))["image"]
        return image


class BSDS300(D.Dataset):
    """
    Reference: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
    """
    def __init__(self, root, regex="*.jpg", train=False, transform=None):
        self.root = root
        self.regex = regex
        self.train = train
        self.sigma = sigma
        self.transform = transform

        if self.train:
            self.image_paths = glob.glob(
                os.path.join(self.root, 'images', 'train', self.regex))
        else:
            self.image_paths = glob.glob(
                os.path.join(self.root, 'images', 'test', self.regex))

        if self.transform is None:
            # Default set of transforms if none are provided
            self.transform = A.Compose([
                A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), p=1),
                ToTensorV2()
            ])

    @staticmethod
    def download(url, destination_dir="./data/bsds300", force=False):
        destination_dir = Path(destination_dir)

        if url is None:
            url = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"

        # Check if dataset has already been downloaded into `destination_dir`
        if destination_dir.is_dir() and not force:
            logger.info(
                f"BSDS300 has already been downloaded into {destination_dir}. Use force=True to download again."
            )
            return

        # Download & Extract
        fname = utils.download_file(url, destination_dir)
        utils.extract_file(fname, destination_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Read path to image
        image_path = self.image_paths[index]
        # Read image
        image = np.asarray(Image.open(image_path).convert('RGB'))
        # Data augmentation
        image = self.transform(image=image)['image']
        return image


class BSDS500(D.Dataset):
    """
    Reference: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html
    """
    def __init__(self, root, regex="*.jpg", train=False, transform=None):
        self.root = root
        self.regex = regex
        self.train = train
        self.sigma = sigma
        self.transform = transform

        if self.train:
            self.image_paths = glob.glob(
                os.path.join(self.root, 'images', 'train', self.regex))
        else:
            self.image_paths = glob.glob(
                os.path.join(self.root, 'images', 'test', self.regex))

        if self.transform is None:
            # Default set of transforms if none are provided
            self.transform = A.Compose([
                A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), p=1),
                ToTensorV2()
            ])

    @staticmethod
    def download(url, destination_dir="./data/bsds500", force=False):
        destination_dir = Path(destination_dir)

        if url is None:
            url = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"

        # Check if dataset has already been downloaded into `destination_dir`
        if destination_dir.is_dir() and not force:
            logger.info(
                f"BSDS500 has already been downloaded to {destination_dir}. Use force=True to download again."
            )
            return

        # Download & Extract
        fname = utils.download_file(url, destination_dir)
        utils.extract_file(fname, destination_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Read path to image
        image_path = self.image_paths[index]
        # Read image
        image = np.asarray(Image.open(image_path).convert('RGB'))
        # Data augmentation
        image = self.transform(image=image)['image']
        return image


class SIDD(D.Dataset):
    def __init__(self):
        pass

    @staticmethod
    def download(urls, destination_dir="data/sidd", force=False):
        # Urls from https://www.eecs.yorku.ca/~kamel/sidd/files/SIDD_URLs_Mirror_2.txt
        if urls is None:
            raise ValueError("Provide a list of urls.")

        destination_dir = Path(destination_dir)
        if destination_dir.is_dir() and not force:
            logger.warning(
                f"{str(destination_dir)} exists. Use force=True to download again."
            )

        for i, url in enumerate(urls):
            fname = utils.download_file(url, destination_dir,
                                        f"Downloading SIDD[{i}]")
            utils.extract_file(fname,
                               ".zip",
                               destination_dir,
                               f"Extracting SIDD[{i}]",
                               remove_extract=True)

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


class DarmstadtNoise(D.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


class LOL(D.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


class ExDark(D.Dataset):
    def __init__(self):
        pass

    @staticmethod
    def download(url, destination_dir, force=False):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


class VV(D.Dataset):
    def __init__(self, root, regex, transform=None):
        self.root = Path(root)
        self.image_paths = list(self.root.glob(regex))
        self.transform = transform
        if self.transform is None:
            # Default set of transforms if none are provided
            self.transform = A.Compose([
                A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), p=1),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = np.asarray(Image.open(image_path))
        image = self.transform(image=image)['image']
        return image


class VIPLowNoise(D.Dataset):
    def __init__(self, root, regex="*.png", transform=None):
        self.root = Path(root)
        self.image_paths = list(self.root.glob(regex))
        self.transform = transform
        if self.transform is None:
            # Default set of transforms if none are provided
            self.transform = A.Compose([
                A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), p=1),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = np.asarray(Image.open(image_path))
        image = self.transform(image=image)['image']
        return image


class DICM(D.Dataset):
    def __init__(self, root, regex, transform=None):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


class LIME(D.Dataset):
    def __init__(self, root, regex, transform=None):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


if __name__ == "__main__":
    # TODO Move these to tests/

    # Test the dataset API
    train_transform = A.Compose([
        A.VerticalFlip(p=0.1),
        A.HorizontalFlip(p=0.6),
        A.ShiftScaleRotate(shift_limit=0.05,
                        scale_limit=0.05,
                        rotate_limit=15,
                        p=1),
        A.Resize(256, 256, interpolation=INTER_LANCZOS4, p=1),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1),
        A.MotionBlur(p=1),
        A.RandomBrightnessContrast(p=1),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), p=1),
        ToTensorV2(),
    ])

    ds = ZeroDceDS("data/train_data/",
                "*.jpg",
                train=True,
                transform=train_transform)
    dl = D.DataLoader(ds, batch_size=8, pin_memory=False, shuffle=False)
    batch = next(iter(dl))
    plt.imshow(batch[3].permute(1, 2, 0))
    plt.show()
    logger.debug(f"{batch[3].max(), batch[3].min()}")
