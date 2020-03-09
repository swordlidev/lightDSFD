import torch
import torch.nn as nn
from operations import OPS, BN_OPS, NORMAL_OPS

import torch.nn.functional as F
import math
# from genotypes import FPN_Genotype

from dataset.config import widerface_640 as cfg

# FPN_Genotype = namedtuple("FPN_Genotype", "Inter_Layer Out_Layer")

BiFPN_PRIMITIVES = [
    # "none",
    # "max_pool_3x3",
    # "avg_pool_3x3",
    "conv_1x1",
    "sep_conv_3x3",
    "sep_conv_5x5",
    "dil_conv_3x3",
    "dil_conv_3x3_3",
    "dil_conv_5x5",
    # 'sep_conv_7x7',
    # "conv_1x3_3x1",
    # "conv_1x5_5x1",
    # 'dconv_3x3',
    # 'conv_1x3',
    # 'conv_3x1',
    # 'conv_1x5',
    # 'conv_5x1',
]


# for retraining the network
class BiFPN_From_Genotype(nn.Module):
    """ Build a FPN cell accroding to its genotype file """

    def __init__(self, genotype, feature_channel=256, weight_node=False, **kwargs):
        """
        :param genotype:
            The Genotype is formatted as follow:
            [
                # for a node
                [
                    # for an operation
                    (prim, number of the front node)
                    # other ops
                    ...
                ]
                # other nodes
                ...
            ]
        :param feature_channel:
        """
        super(BiFPN_From_Genotype, self).__init__()

        bn = True

        # ops = NORMAL_OPS
        if bn:
            ops = BN_OPS
            print("Retrain with BN - FPN.")
        else:
            ops = OPS
            print("Retrain without BN - FPN.")

        print(ops.keys())

        self.feature_channel = feature_channel

        self.genotype = genotype
        self.node_weights_enable = weight_node

        # sharing the same structure of genotype
        [
            self.conv1_td,
            self.conv1,
            self.conv2_td,
            self.conv2_du,
            self.conv2,
            self.conv3_td,
            self.conv3_du,
            self.conv3,
            self.conv4_td,
            self.conv4_du,
            self.conv4,
            self.conv5_td,
            self.conv5_du,
            self.conv5,
            self.conv6_du,
            self.conv6,
        ] = [ops[prim](feature_channel, 1, True) for node in self.genotype for prim, _ in node]

        [self.w1, self.w2, self.w3, self.w4, self.w5, self.w6] = [
            nn.Parameter(1e-3 * torch.randn(len(node))) for node in self.genotype
        ]

        self.out_layers = nn.ModuleList([nn.Conv2d(feature_channel, feature_channel, 1, 1, 0) for _ in range(6)])

    def upsample_as(self, x, y):
        return F.interpolate(x, size=y.shape[2:], mode="bilinear", align_corners=True)

    def max_pool(self, x):
        return F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

    def forward(self, sources):
        """
        forward function
        """
        f1, f2, f3, f4, f5, f6 = sources

        # top-down path
        f6_td = f6
        f5_td = self.conv5_td(self.upsample_as(f6_td, f5) * f5)
        f4_td = self.conv4_td(self.upsample_as(f5_td, f4) * f4)
        f3_td = self.conv3_td(self.upsample_as(f4_td, f3) * f3)
        f2_td = self.conv2_td(self.upsample_as(f3_td, f2) * f2)
        f1_td = self.conv1_td(self.upsample_as(f2_td, f1) * f1)

        # bottom-up path
        f1_du = f1
        f2_du = self.conv2_du(self.max_pool(f1_du) * f2)
        f3_du = self.conv3_du(self.max_pool(f2_du) * f3)
        f4_du = self.conv4_du(self.max_pool(f3_du) * f4)
        f5_du = self.conv5_du(self.max_pool(f4_du) * f5)
        f6_du = self.conv6_du(self.max_pool(f5_du) * f6)

        # output
        f1_out = self.conv1(f1)
        f2_out = self.conv2(f2)
        f3_out = self.conv3(f3)
        f4_out = self.conv4(f4)
        f5_out = self.conv5(f5)
        f6_out = self.conv6(f6)

        return [
            self.out_layers[0](
                (torch.stack([f1_td, f1_out], dim=-1) * F.softmax(self.w1, dim=0)).sum(dim=-1, keepdim=False)
            ),
            self.out_layers[1](
                (torch.stack([f2_td, f2_du, f2_out], dim=-1) * F.softmax(self.w2, dim=0)).sum(dim=-1, keepdim=False)
            ),
            self.out_layers[2](
                (torch.stack([f3_td, f3_du, f3_out], dim=-1) * F.softmax(self.w3, dim=0)).sum(dim=-1, keepdim=False)
            ),
            self.out_layers[3](
                (torch.stack([f4_td, f4_du, f4_out], dim=-1) * F.softmax(self.w4, dim=0)).sum(dim=-1, keepdim=False)
            ),
            self.out_layers[4](
                (torch.stack([f5_td, f5_du, f5_out], dim=-1) * F.softmax(self.w5, dim=0)).sum(dim=-1, keepdim=False)
            ),
            self.out_layers[5](
                (torch.stack([f6_du, f6_out], dim=-1) * F.softmax(self.w6, dim=0)).sum(dim=-1, keepdim=False)
            ),
        ]


class BiFPN_Neck_From_Genotype(nn.Module):
    """ FPN_Neck from genotype file """

    def __init__(
        self, genotype, in_channels=256, feature_size=256, weight_node=False, fpn_layers=1,
    ):
        super(BiFPN_Neck_From_Genotype, self).__init__()

        genotype = genotype.Inter_Layer

        self.fpn_layers = fpn_layers
        if fpn_layers == 1:
            self.layers = BiFPN_From_Genotype(genotype, feature_channel=feature_size, weight_node=weight_node,)
        else:
            self.layers = nn.ModuleList()

            for i in range(fpn_layers):
                self.layers.append(BiFPN_From_Genotype(genotype, feature_channel=feature_size, weight_node=weight_node))

    def forward(self, source):
        """ forward function """

        if self.fpn_layers == 1:
            return self.layers(source)

        else:
            for layer in self.layers:
                source = layer(source)
            return source

