import os
import glob
import numpy as np
import torch
from torch.utils import data as D
import albumentations as A
from albumentations.pytorch import ToTensorV2

import pytest
from nightsight import data

class TestZeroDceDS:
    tf = A.Compose([
        A.Resize(256, 256, interpolation=4, p=1),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2()
    ])

    root = "data/train_data"
    extn = "*.jpg"
    ds = data.ZeroDceDS(root, extn, train=True, transform=tf)
    dl = D.DataLoader(ds, batch_size=4, pin_memory=False, shuffle=False)

    def test_image_collection(self):
        nimages = len(glob.glob(os.path.join(self.root, self.extn)))
        assert len(self.ds) == nimages

    @pytest.mark.slow
    def test_output_size(self):
        for image in self.ds:
            assert tuple(image.shape) == (3, 256, 256)

    @pytest.mark.slow
    def test_output_range(self):
        for image in self.ds:
            assert torch.le(torch.max(image), torch.Tensor([1.0]))
            assert torch.ge(torch.min(image), torch.Tensor([0.0]))


@pytest.mark.skip(reason="Tired")
class TestGoogleOpenImagesDS:
    @pytest.mark.slow
    def test_download(self):
        pass

    def test_image_collection(self):
        pass

    @pytest.mark.slow
    def test_output_size(self):
        pass

    @pytest.mark.slow
    def test_output_range(self):
        pass


@pytest.mark.skip(reason="Tired")
class TestBSDS300:
    @pytest.mark.slow
    def test_download(self):
        pass

    def test_image_collection(self):
        pass

    @pytest.mark.slow
    def test_output_size(self):
        pass

    @pytest.mark.slow
    def test_output_range(self):
        pass


@pytest.mark.skip(reason="Tired")
class TestGoogleOpenImagesDS:
    @pytest.mark.slow
    def test_download(self):
        pass

    def test_image_collection(self):
        pass

    @pytest.mark.slow
    def test_output_size(self):
        pass

    @pytest.mark.slow
    def test_output_range(self):
        pass


@pytest.mark.skip(reason="WIP")
class TestSIDD:
    @pytest.mark.slow
    def test_download(self):
        pass

    def test_image_collection(self):
        pass

    @pytest.mark.slow
    def test_output_size(self):
        pass

    @pytest.mark.slow
    def test_output_range(self):
        pass


@pytest.mark.skip(reason="WIP")
class TestDarmstadtNoise:
    @pytest.mark.slow
    def test_download(self):
        pass

    def test_image_collection(self):
        pass

    @pytest.mark.slow
    def test_output_size(self):
        pass

    @pytest.mark.slow
    def test_output_range(self):
        pass


@pytest.mark.skip(reason="WIP")
class TestLOL:
    @pytest.mark.slow
    def test_download(self):
        pass

    def test_image_collection(self):
        pass

    @pytest.mark.slow
    def test_output_size(self):
        pass

    @pytest.mark.slow
    def test_output_range(self):
        pass


@pytest.mark.skip(reason="WIP")
class TestExDark:
    @pytest.mark.slow
    def test_download(self):
        pass

    def test_image_collection(self):
        pass

    @pytest.mark.slow
    def test_output_size(self):
        pass

    @pytest.mark.slow
    def test_output_range(self):
        pass


@pytest.mark.skip(reason="WIP")
class TestVV:
    @pytest.mark.slow
    def test_download(self):
        pass

    def test_image_collection(self):
        pass

    @pytest.mark.slow
    def test_output_size(self):
        pass

    @pytest.mark.slow
    def test_output_range(self):
        pass


@pytest.mark.skip(reason="WIP")
class TestVIPLowNoise:
    @pytest.mark.slow
    def test_download(self):
        pass

    def test_image_collection(self):
        pass

    @pytest.mark.slow
    def test_output_size(self):
        pass

    @pytest.mark.slow
    def test_output_range(self):
        pass


@pytest.mark.skip(reason="WIP")
class TestDICM:
    @pytest.mark.slow
    def test_download(self):
        pass

    def test_image_collection(self):
        pass

    @pytest.mark.slow
    def test_output_size(self):
        pass

    @pytest.mark.slow
    def test_output_range(self):
        pass


@pytest.mark.skip(reason="WIP")
class TestDICM:
    @pytest.mark.slow
    def test_download(self):
        pass

    def test_image_collection(self):
        pass

    @pytest.mark.slow
    def test_output_size(self):
        pass

    @pytest.mark.slow
    def test_output_range(self):
        pass


@pytest.mark.skip(reason="WIP")
class TestLIME:
    @pytest.mark.slow
    def test_download(self):
        pass

    def test_image_collection(self):
        pass

    @pytest.mark.slow
    def test_output_size(self):
        pass

    @pytest.mark.slow
    def test_output_range(self):
        pass
