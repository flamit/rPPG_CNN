import torch.nn as nn
import torch
import torch.nn.functional as F


class RGAS(nn.Module):
    def __init__(self, inplanes, h, w, s, affinity_out, s2):
        super(RGAS, self).__init__()
        self.s = s
        self.conv1 = nn.Conv2d(inplanes, inplanes // s, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(inplanes // s)
        self.conv2 = nn.Conv2d(inplanes, inplanes // s, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(inplanes // s)
        self.relu = nn.ReLU()

        self.conv3 = nn.Conv2d(inplanes, inplanes // s, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(inplanes // s)
        self.conv4 = nn.Conv2d(h * w, affinity_out, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(affinity_out)

        current_feats = 1 + affinity_out * 2
        self.conv5 = nn.Conv2d(current_feats, current_feats // s2, kernel_size=1)
        self.bn5 = nn.BatchNorm2d(current_feats // s2)
        self.conv6 = nn.Conv2d(current_feats // s2, 1, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        # Embed x using the two functions:
        theta = self.relu(self.bn1(self.conv1(x))).view(-1, c // self.s, h * w)
        phi = self.relu(self.bn2(self.conv2(x))).view(-1, c // self.s, h * w)
        affinity = torch.zeros(size=[b, h * w, h * w])
        # Calculate affinity as convolution over each batch for faster computation:
        for batch in range(b):
            kernel = phi[batch].permute(0, 1).view(h * w, c // self.s, 1, 1)
            r = F.conv2d(theta[batch].view(1, c // self.s, h, w), kernel).view(h * w, h * w)
            affinity[batch] = r
        # Take out affinity row wise and column wise and reshape to image dimensions.
        affinity_a = affinity.view(b, -1, h, w)
        affinity_b = affinity.permute(0, 2, 1).view(b, -1, h, w)

        # Embed x and affinity so that they are in the same feature space.
        x_embed = self.relu(self.bn3(self.conv3(x))).mean(dim=1, keepdim=True)  # Mean pool over channels
        affinitya_embed = self.relu(self.conv4(affinity_a))
        affinityb_embed = self.relu(self.conv4(affinity_b))
        # Calculate the y tensor as the concatenation of affinity and x
        y = torch.cat([x_embed, affinitya_embed, affinityb_embed], dim=1)
        # Calculate attention
        a = self.sig(self.conv6(self.relu(self.bn5(self.conv5(y)))))

        return x * a


# TODO Channel attention part, then combine and add to resnet:
class RGAC(nn.Module):
    def __init__(self, inplanes, h, w, s, affinity_out, s2):
        super(RGAC, self).__init__()
        pass