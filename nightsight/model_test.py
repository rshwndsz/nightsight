import pytest
import torch
from nightsight import model


class TestEnhanceNetNoPoolCPU:
    net = model.EnhanceNetNoPool()
    inputs = torch.rand(4, 3, 256, 256)
    ei_1, ei, A = net(inputs)
    ei_1, ei, A = ei_1.detach(), ei.detach(), A.detach()

    def test_enhanced_images_shape(self):
        assert self.ei.shape == self.inputs.shape

    def test_enhanced_images_range(self):
        assert torch.le(torch.max(self.ei), torch.tensor([1.0]))
        assert torch.ge(torch.min(self.ei), torch.tensor([0.0]))


# @pytest.mark.skipif(condition=(not torch.cuda.is_available()), reason="No GPU")
# class TestEnhanceNetNoPoolGPU:
#     net = model.EnhanceNetNoPool().to('cuda:0')
#     inputs = torch.rand(4, 3, 256, 256).to('cuda:0')
#     ei_1, ei, A = net(inputs).to('cuda:0')
#     ei_1, ei, A = ei_1.detach(), ei.detach(), A.detach()

#     def test_enhanced_images_shape(self):
#         assert self.ei.shape == self.inputs.shape

#     def test_enhanced_images_range(self):
#         assert torch.le(torch.max(self.ei), torch.tensor([1.0]))
#         assert torch.ge(torch.min(self.ei), torch.tensor([0.0]))