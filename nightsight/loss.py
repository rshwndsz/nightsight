import torch
from torch import nn
from torch.nn import functional as F

class LSpaLoss(nn.Module):
    """Spatial consistency loss

        The spatial consistency loss encourages spatial coherence of the
        enhanced image through preserving the difference of neighboring
        regions between the input image and its enhanced version.
    """
    def __init__(self, type_ref):
        super(LSpaLoss, self).__init__()
        kernel_l = torch.FloatTensor(
            [[0, 0, 0], [-1, 1, 0],
             [0, 0, 0]]).type_as(type_ref).unsqueeze(0).unsqueeze(0)
        kernel_r = torch.FloatTensor(
            [[0, 0, 0], [0, 1, -1],
             [0, 0, 0]]).type_as(type_ref).unsqueeze(0).unsqueeze(0)
        kernel_u = torch.FloatTensor(
            [[0, -1, 0], [0, 1, 0],
             [0, 0, 0]]).type_as(type_ref).unsqueeze(0).unsqueeze(0)
        kernel_d = torch.FloatTensor(
            [[0, 0, 0], [0, 1, 0],
             [0, -1, 0]]).type_as(type_ref).unsqueeze(0).unsqueeze(0)
        self.weight_l = nn.Parameter(data=kernel_l, requires_grad=False)
        self.weight_r = nn.Parameter(data=kernel_r, requires_grad=False)
        self.weight_u = nn.Parameter(data=kernel_u, requires_grad=False)
        self.weight_d = nn.Parameter(data=kernel_d, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        b, c, h, w = org.shape

        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        weight_diff = torch.max(
            torch.FloatTensor([1]).type_as(org) +
            10000 * torch.min(org_pool - torch.FloatTensor([0.3]).type_as(org),
                              torch.FloatTensor([0]).type_as(org)),
            torch.FloatTensor([0.5]).type_as(org))
        E_1 = torch.mul(
            torch.sign(enhance_pool - torch.FloatTensor([0.5]).type_as(org)),
            enhance_pool - org_pool)

        D_org_l = F.conv2d(org_pool, self.weight_l, padding=1)
        D_org_r = F.conv2d(org_pool, self.weight_r, padding=1)
        D_org_u = F.conv2d(org_pool, self.weight_u, padding=1)
        D_org_d = F.conv2d(org_pool, self.weight_d, padding=1)

        D_enhance_l = F.conv2d(enhance_pool, self.weight_l, padding=1)
        D_enhance_r = F.conv2d(enhance_pool, self.weight_r, padding=1)
        D_enhance_u = F.conv2d(enhance_pool, self.weight_u, padding=1)
        D_enhance_d = F.conv2d(enhance_pool, self.weight_d, padding=1)

        D_l = torch.pow(D_org_l - D_enhance_l, 2)
        D_r = torch.pow(D_org_r - D_enhance_r, 2)
        D_u = torch.pow(D_org_u - D_enhance_u, 2)
        D_d = torch.pow(D_org_d - D_enhance_d, 2)

        E = (D_l + D_r + D_u + D_d)
        return E


class LExpLoss(nn.Module):
    """Exposure control loss

    To restrain under/over-exposed regions.
    The exposure control loss measures the distnce between the average intensity
    value of a local region to the well-exposedness level E (set as 0.6)
    """
    def __init__(self, patch_size, mean_val):
        super(LExpLoss, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        b, c, h, w = x.shape
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)

        d = torch.mean(
            torch.pow(mean - torch.Tensor([self.mean_val]).type_as(x), 2))
        return d


class LColorLoss(nn.Module):
    """Color constancy loss

    Follows the Gray-World color constancy hypothesis that color in each sensor
    averages to gray over the entire image.
    The color constancy loss is designed to correct the potential color deviations
    in the enhanced image and also build the relations among the 3 adjusted channels.
    """
    def __init__(self):
        super(LColorLoss, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)

        k = torch.pow(
            torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)
        return k


class LTVLoss(nn.Module):
    """Illumination smoothness loss

    This loss is added to each curve parameter map to preserve
    the monotonicity relations between neighboring pixels.
    """
    def __init__(self, TVLoss_weight=1):
        super(LTVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()

        return self.TVLoss_weight * 2 * (h_tv / count_h +
                                         w_tv / count_w) / batch_size


class SaLoss(nn.Module):
    def _init__(self):
        super(SaLoss, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape
        r, g, b = torch.split(x, 1, dim=1)
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = r - mr
        Dg = g - mg
        Db = b - mb
        k = torch.pow(
            torch.pow(Dr, 2) + torch.pow(Db, 2) + torch.pow(Dg, 2), 0.5)
        k = torch.mean(k)
        return k


class PerceptionLoss(nn.Module):
    def __init__(self):
        super(PerceptionLoss, self).__init__()
        features = tv.models.vgg.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # Don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        return h_relu_4_3
