import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.config import widerface_640 as cfg
# from layers.DCNv2 import DCN

RELU_FIRST = True

OPS = {
    "none": lambda C, stride, affine: Zero(stride),
    "avg_pool_3x3": lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    "max_pool_3x3": lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    "skip_connect": lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    "sep_conv_3x3": lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    "sep_conv_5x5": lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    "sep_conv_7x7": lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    "dil_conv_3x3": lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    "dil_conv_3x3_3": lambda C, stride, affine: DilConv(C, C, 3, stride, 3, 3, affine=affine),
    "dil_conv_5x5": lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    "conv_1x3_3x1": lambda C, stride, affine: RfeConv(C, C, 3, stride, 1, affine=affine),
    "conv_1x5_5x1": lambda C, stride, affine: RfeConv(C, C, 5, stride, 2, affine=affine),
    

    # "dconv_3x3": lambda C, stride, affine: D_Conv(C, C, 3, 1, affine=affine, bn=False),
    "conv_1x3": lambda C, stride, affine: Normal_Relu_Conv(C, C, kernel_size=(1,3), stride=stride, padding=(0,1), bn=False),
    "conv_3x1": lambda C, stride, affine: Normal_Relu_Conv(C, C, kernel_size=(3,1), stride=stride, padding=(1,0), bn=False),
    "conv_1x5": lambda C, stride, affine: Normal_Relu_Conv(C, C, kernel_size=(1,5), stride=stride, padding=(0,2), bn=False),
    "conv_5x1": lambda C, stride, affine: Normal_Relu_Conv(C, C, kernel_size=(5,1), stride=stride, padding=(2,0), bn=False),
    "conv_1x1": lambda C, stride, affine: Normal_Relu_Conv(C, C, kernel_size=1, stride=1, padding=0, bn=False),
}  # black: disable

BN_OPS = {
    "none": lambda C, stride, affine: Zero(stride),
    "avg_pool_3x3": lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    "max_pool_3x3": lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    "skip_connect": lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    "sep_conv_3x3": lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine, bn=True),
    "sep_conv_5x5": lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine, bn=True),
    "sep_conv_7x7": lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine, bn=True),
    "dil_conv_3x3": lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine, bn=True),
    "dil_conv_3x3_3": lambda C, stride, affine: DilConv(C, C, 3, stride, 3, 3, affine=affine, bn=True),
    "dil_conv_5x5": lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine, bn=True),
    "conv_1x3_3x1": lambda C, stride, affine: RfeConv(C, C, 3, stride, 1, affine=affine, bn=True),
    "conv_1x5_5x1": lambda C, stride, affine: RfeConv(C, C, 5, stride, 2, affine=affine, bn=True),
    # "dconv_3x3": lambda C, stride, affine: D_Conv(C, C, 3, 1, affine=affine, bn=True),
    "conv_1x3": lambda C, stride, affine: Normal_Relu_Conv(C, C, kernel_size=(1,3), stride=stride, padding=(0,1), bn=True),
    "conv_3x1": lambda C, stride, affine: Normal_Relu_Conv(C, C, kernel_size=(3,1), stride=stride, padding=(1,0), bn=True),
    "conv_1x5": lambda C, stride, affine: Normal_Relu_Conv(C, C, kernel_size=(1,5), stride=stride, padding=(0,2), bn=True),
    "conv_5x1": lambda C, stride, affine: Normal_Relu_Conv(C, C, kernel_size=(5,1), stride=stride, padding=(2,0), bn=True),
    "conv_1x1": lambda C, stride, affine: Normal_Relu_Conv(C, C, kernel_size=1, stride=1, padding=0, bn=True),
}

NORMAL_OPS = {
    "none": lambda C, stride, affine: Zero(stride),
    "avg_pool_3x3": lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    "max_pool_3x3": lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    "skip_connect": lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    "sep_conv_3x3": lambda C, stride, affine: Normal_Relu_Conv(C, C, kernel_size=3, stride=stride, padding=1, bn=True),
    "sep_conv_5x5": lambda C, stride, affine: Normal_Relu_Conv(C, C, kernel_size=5, stride=stride, padding=1, bn=True),
    "sep_conv_7x7": lambda C, stride, affine: Normal_Relu_Conv(C, C, kernel_size=7, stride=stride, padding=1, bn=True),
    "dil_conv_3x3": lambda C, stride, affine: Normal_Relu_Conv(C, C, kernel_size=3, stride=stride, padding=2, bn=True, dilation=2),
    "dil_conv_3x3_3": lambda C, stride, affine: Normal_Relu_Conv(C, C, kernel_size=3, stride=stride, padding=3, bn=True, dilation=3),
    "dil_conv_5x5": lambda C, stride, affine: Normal_Relu_Conv(C, C, kernel_size=5, stride=stride, padding=4, bn=True, dilation=2),
    "conv_1x3_3x1": lambda C, stride, affine: RfeConv(C, C, 3, stride, 1, affine=affine, bn=True),
    "conv_1x5_5x1": lambda C, stride, affine: RfeConv(C, C, 5, stride, 2, affine=affine, bn=True),
    # "dconv_3x3": lambda C, stride, affine: D_Conv(C, C, 3, 1, affine=affine, bn=True),
    "conv_1x3": lambda C, stride, affine: Normal_Relu_Conv(C, C, kernel_size=(1,3), stride=stride, padding=(0,1), bn=True),
    "conv_3x1": lambda C, stride, affine: Normal_Relu_Conv(C, C, kernel_size=(3,1), stride=stride, padding=(1,0), bn=True),
    "conv_1x5": lambda C, stride, affine: Normal_Relu_Conv(C, C, kernel_size=(1,5), stride=stride, padding=(0,2), bn=True),
    "conv_5x1": lambda C, stride, affine: Normal_Relu_Conv(C, C, kernel_size=(5,1), stride=stride, padding=(2,0), bn=True),
}


class Normal_Relu_Conv(nn.Module):
    def __init__(self, C_in, C_out, affine=True, bn=False, **kwargs):
        super(Normal_Relu_Conv, self).__init__()
        if not bn:
            op = nn.Sequential(
                # nn.ReLU(),
                nn.Conv2d(C_in, C_in, bias=True, **kwargs),
            )
        else:
            if cfg['GN']:
                bn_layer = nn.GroupNorm(32, C_out)
            elif cfg["syncBN"]:
                bn_layer = nn.SyncBatchNorm(C_out)
            else:
                bn_layer = nn.BatchNorm2d(C_out)
                
            op = nn.Sequential(
                # nn.ReLU(),
                nn.Conv2d(C_in, C_in, bias=False, **kwargs),
                bn_layer,
            )
        
        if RELU_FIRST:
            self.op = nn.Sequential()
            self.op.add_module('0', nn.ReLU())
            for i in range(1, len(op)+1):
                self.op.add_module(str(i), op[i-1])
        else:
            self.op = op
            self.op.add_module(str(len(op)), nn.ReLU())
        # self.op = op
            
    def forward(self, x):
        return self.op(x)



class _GumbelSoftMax(torch.autograd.Function):
    """
    implementing the MixedOp, but carried out in a different way as DARTS

    DARTS adds all operations together, then select the maximal one to construct the final network,
    however, during the late process, more weights are assigned to the None, this is unreasonable under the
    circumstance that per operation has the unsure number of inputs.

    Thus, we modifies the original DARTS by applying way in GDAS to test.

    This class aims to compute the gradients by ourself.
    """

    @staticmethod
    def forward(ctx, weights):
        weights_norm = F.softmax(weights, dim=-1)
        ctx.saved_for_backward = weights_norm

        # select the max one
        mask = torch.zeros_like(weights_norm).to(weights.device)
        _, idx = weights_norm.topk(dim=-1, k=1, largest=True)
        mask[idx] = 1.0
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        gumbel_norm = ctx.saved_for_backward
        return gumbel_norm * (1 - gumbel_norm) * grad_output * gumbel_norm.shape[0]


class GumbelSoftMax(nn.Module):
    def __init__(self):
        super(GumbelSoftMax, self).__init__()

    def forward(self, weights, temp_coeff=1.0):
        gumbel = -1e-3 * torch.log(-torch.log(torch.rand_like(weights))).to(weights.device)
        weights = _GumbelSoftMax.apply((weights + gumbel) / temp_coeff)
        return weights


# class D_Conv(nn.Module):
#     """ Deformable Conv V2 """

#     def __init__(self, C_in, C_out, kernel_size, padding, affine=True, bn=False):
#         super(D_Conv, self).__init__()
#         if bn:
#             if cfg["syncBN"]:
#                 bn_layer = nn.SyncBatchNorm(C_out)
#             else:
#                 bn_layer = nn.BatchNorm2d(C_out)
#             self.op = nn.Sequential(
#                 nn.ReLU(inplace=False),
#                 DCN(
#                     C_in, C_in, kernel_size=kernel_size, padding=padding, stride=1, deformable_groups=C_in, groups=C_in
#                 ),
#                 nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
#                 bn_layer,
#             )
#         else:
#             self.op = nn.Sequential(
#                 nn.ReLU(inplace=False),
#                 DCN(
#                     C_in, C_in, kernel_size=kernel_size, padding=padding, stride=1, deformable_groups=C_in, groups=C_in
#                 ),
#                 nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=True),
#             )

#     def forward(self, x):
#         return self.op(x)


class RfeConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, bn=False):
        super(RfeConv, self).__init__()
        if not bn:
            op = nn.Sequential(
                # nn.ReLU(),
                nn.Conv2d(
                    C_in,
                    C_in,
                    kernel_size=(1, kernel_size),
                    stride=(1, stride),
                    padding=(0, padding),
                    groups=C_in,
                    bias=True,
                ),
                # nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=True),
                nn.Conv2d(
                    C_in,
                    C_in,
                    kernel_size=(kernel_size, 1),
                    stride=(stride, 1),
                    padding=(padding, 0),
                    groups=C_in,
                    bias=True,
                ),
                nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=True),
            )
        else:
            if cfg['GN']:
                bn_layer = nn.GroupNorm(32, C_out)
            elif cfg["syncBN"]:
                bn_layer = nn.SyncBatchNorm(C_out)
            else:
                bn_layer = nn.BatchNorm2d(C_out)
                
            op = nn.Sequential(
                # nn.ReLU(),
                nn.Conv2d(
                    C_in,
                    C_in,
                    kernel_size=(1, kernel_size),
                    stride=(1, stride),
                    padding=(0, padding),
                    groups=C_in,
                    bias=True,
                ),
                nn.Conv2d(
                    C_in,
                    C_in,
                    kernel_size=(kernel_size, 1),
                    stride=(stride, 1),
                    padding=(padding, 0),
                    groups=C_in,
                    bias=True,
                ),
                nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=True),
                bn_layer,
            )

        if RELU_FIRST:
            self.op = nn.Sequential()
            self.op.add_module('0', nn.ReLU())
            for i in range(1, len(op)+1):
                self.op.add_module(str(i), op[i-1])
        else:
            self.op = op
            self.op.add_module(str(len(op)), nn.ReLU())
        # self.op = op

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True, bn=False):
        super(DilConv, self).__init__()
        if not bn:
            op = nn.Sequential(
                # nn.ReLU(),
                nn.Conv2d(
                    C_in,
                    C_in,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=C_in,
                    bias=True,
                ),
                nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=True),
            )
        else:
            if cfg['GN']:
                bn_layer = nn.GroupNorm(32, C_out)
            elif cfg["syncBN"]:
                bn_layer = nn.SyncBatchNorm(C_out)
            else:
                bn_layer = nn.BatchNorm2d(C_out)
            
            op = nn.Sequential(
                # nn.ReLU(),
                nn.Conv2d(
                    C_in,
                    C_in,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=C_in,
                    bias=False,
                ),
                nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
                bn_layer,
            )

        if RELU_FIRST:
            self.op = nn.Sequential()
            self.op.add_module('0', nn.ReLU())
            for i in range(1, len(op)+1):
                self.op.add_module(str(i), op[i-1])
        else:
            self.op = op
            self.op.add_module(str(len(op)), nn.ReLU())
        # self.op = op
            
    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, bn=False):
        super(SepConv, self).__init__()
        if not bn:
            op = nn.Sequential(
                # nn.ReLU(),
                nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=True,),
                nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=True),
            )
        else:
            if cfg['GN']:
                bn_layer = nn.GroupNorm(32, C_out)
            elif cfg["syncBN"]:
                bn_layer = nn.SyncBatchNorm(C_out)
            else:
                bn_layer = nn.BatchNorm2d(C_out)
                
            op = nn.Sequential(
                # nn.ReLU(),
                nn.Conv2d(
                    C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False,
                ),
                nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
                bn_layer,
            )

        if RELU_FIRST:
            self.op = nn.Sequential(nn.ReLU())
            # self.op.add_module('0', nn.ReLU())
            for i in range(1, len(op)+1):
                self.op.add_module(str(i), op[i-1])
        else:
            self.op = op
            self.op.add_module(str(len(op)), nn.ReLU())
        # self.op = op
            
    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.0)
        return x[:, :, :: self.stride, :: self.stride].mul(0.0)
