import pytest
import torch
from nightsight import loss


def test_LSpaLoss():
    inputs = torch.rand(4, 3, 256, 256)
    enhanced_images = torch.rand(4, 3, 256, 256)
    L_spa = loss.LSpaLoss(type_ref=inputs)
    loss_spa = torch.mean(L_spa(enhanced_images, inputs))
    assert (not torch.isnan(loss_spa))


def test_LExpLoss():
    enhanced_images = torch.rand(4, 3, 256, 256)
    L_exp = loss.LExpLoss(16, 0.6)
    loss_exp = torch.mean(L_exp(enhanced_images))
    assert (not torch.isnan(loss_exp))


def test_LColorLoss():
    enhanced_images = torch.rand(4, 3, 256, 256)
    L_color = loss.LColorLoss()
    loss_col = torch.mean(L_color(enhanced_images))
    assert (not torch.isnan(loss_col))


def test_LTVLoss():
    A = torch.rand(4, 24, 256, 256)
    L_tv = loss.LTVLoss()
    loss_tv = L_tv(A)
    assert (not torch.isnan(loss_tv))


@pytest.mark.skip(reason="Not used")
def test_SaLoss():
    pass


@pytest.mark.skip(reason="Not used")
def test_PerceptionLoss():
    pass
