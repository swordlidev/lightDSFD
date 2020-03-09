
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck, BasicBlock
from operations import *
from collections import defaultdict
import math
import pdb
import torchvision
# import genotypes
import time
from dataset import *
from layers import *
from bi_fpn import BiFPN_PRIMITIVES, BiFPN_Neck_From_Genotype
from layers.box_utils import match_anchors, decode, encode
import numbers
from torchvision.ops import roi_align
from collections import OrderedDict

if torch.__version__ == "1.1.0":
    dtype = torch.uint8
else:
    dtype = torch.bool

cfg = widerface_640

base = 1

# print("base: ", base)
mo = cfg["max_out"]

fpn = cfg["feature_pyramid_network"]
search_fpn = cfg["search_feature_pyramid_network"]
use_searched_fpn = cfg["use_searched_feature_pyramid_network"]

bifpn = cfg["bidirectional_feature_pyramid_network"]

cpm = cfg["context_predict_module"]
search_cpm = cfg["search_context_predict_module"]
use_searched_cpm = False

mio = cfg["max_in_out"]
imio = cfg["improved_max_in_out"]

pa = cfg["pyramid_anchor"]
backbone = cfg["backbone"]
bup = cfg["bottom_up_path"]
refine = cfg["refinedet"]
edge_norm = cfg["edge_normalization"]
groups = cfg["groups"]

cross_stack = cfg["cross_stack"]

fpn_cpm_channel = cfg["fpn_cpm_channel"]
stack_convs = cfg["stack_convs"]

margin_loss = cfg["margin_loss_type"]
use_sigmoid = cfg["GHM"]

# whether to enable the STC/STR module
STC_STR = cfg["STC_STR"]
auxiliary_refine = False

import logging

# logging.info("fpn_cpm_channel: %s", fpn_cpm_channel)

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        
        if cfg['GN']:
            self.bn = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-5)
        elif cfg['syncBN']:
            self.bn = nn.SyncBatchNorm(out_channels, eps=1e-5)
        else:
            self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Network(nn.Module):
    def __init__(
        self,
        C,
        num_classes,
        criterion,
        phase,
        search,
        inter_node=3,
        args=None,
        auxiliary_loss=False,  # whether to enable auxiliary loss
        searched_fpn_genotype=None,
        searched_cpm_genotype=None,
        weight_node=False,
        weight_channel=False,
        fusion_multiple=False,
        with_attention=False,
        fpn_layers=1,
        cpm_layers=1,
        layers=1,
        residual_learning=False,
        gumbel_trick=False,
        **kwargs
    ):
        super(Network, self).__init__()

        self._C = C
        self.num_classes = num_classes

        self._criterion = criterion
        self._inter_node = inter_node
        self.fpn_layers = fpn_layers
        self.cpm_layers = cpm_layers
        self.num_layers = layers
        self.weight_node = weight_node

        self.search = search

        self.cfg = cfg
        self.phase = phase

        self.args = args

        self.auxiliary_loss = auxiliary_loss  # whether to enable auxiliary loss
        self.gumbel_trick = gumbel_trick

        if self.search:
            raise ValueError("search mode is not supported.")

        self.searched_fpn_genotype = searched_fpn_genotype

        self.searched_cpm_genotype = searched_cpm_genotype
        
        if cfg['GN']:
            kwargs["norm_layer"] = lambda x: nn.GroupNorm(num_groups=32, num_channels=x)
        elif cfg['syncBN']:
            kwargs["norm_layer"] = nn.SyncBatchNorm
        else:
            kwargs["norm_layer"] = None


        if backbone == 'resnet18':
            try:
                resnet = torchvision.models.resnet18(pretrained=False, **kwargs)
                # resnet.load_state_dict(torch.load("./data/WIDERFace/pretrained_model/resnet18-5c106cde.pth"))
            except Exception as e:
                print(e)
                resnet = torchvision.models.resnet18(pretrained=False, **kwargs)
                # resnet.load_state_dict(torch.load("/data/home/aalenzhang/.cache/torch/checkpoints/resnet18-5c106cde.pth"))
            fpn_in = [64, 128, 256, 512, 256, 256]
            
        else:
            raise ValueError("Backbone {} is not supported.".format(backbone))

        self.layer5 = nn.Sequential(
            *[
                BasicConv2d(fpn_in[-3], fpn_in[-3]//4, kernel_size=1, stride=1, padding=0),
                BasicConv2d(fpn_in[-3]//4, fpn_in[-2], kernel_size=3, stride=2, padding=1),
            ]
        )
        self.layer6 = nn.Sequential(
            *[
                BasicConv2d(fpn_in[-2], max(fpn_in[-2]//4, 128), kernel_size=1, stride=1, padding=0),
                BasicConv2d(max(fpn_in[-2]//4, 128), fpn_in[-1], kernel_size=3, stride=2, padding=1),
            ]
        )

        self.stem_layer = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        
        self.layer1 = nn.Sequential(resnet.layer1)
        self.layer2 = nn.Sequential(resnet.layer2)
        self.layer3 = nn.Sequential(resnet.layer3)
        self.layer4 = nn.Sequential(resnet.layer4)
        
        
        self.fpn_in = fpn_in
        # cpm_in = [256, 256, 256, 256, 256, 256]
        cpm_in = [fpn_cpm_channel] * 6
        fpn_channel = fpn_cpm_channel
        cpm_channels = fpn_cpm_channel

        # output_channels = [ic*4 for ic in cpm_in]
        output_channels = cpm_in
        
        # FPN and CPM modules
        self.FPN_CPM()
        # multibox head
        self.multibox(output_channels, cfg["mbox"], num_classes)
        
        # other parameters
        if self.search:
            self._initialize_alphas()

        if self.phase == "test":
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, cfg["num_thresh"], cfg["conf_thresh"], cfg["nms_thresh"])
        
        self.cfg["feature_maps"] = [[i, i] for i in self.cfg["feature_maps"]]
        self.cfg["min_dim"] = [self.cfg["min_dim"], self.cfg["min_dim"]]
        
        if phase == "train":
            if pa:
                self.face_priors = self.init_priors(self.cfg)
                self.head_priors = self.init_priors(
                    self.cfg, min_size=cfg["min_sizes"][:-1], max_size=cfg["max_sizes"][:-1],
                )
                self.body_priors = self.init_priors(
                    self.cfg, min_size=cfg["min_sizes"][:-2], max_size=cfg["max_sizes"][:-2],
                )
                # ####################### aalenzhang
            # if cfg["auxiliary_classify"]:
            
            else:
                self.priors = self.init_priors(self.cfg)


    
    def FPN_CPM(self):
        fpn_channel = fpn_cpm_channel
        cpm_in = [fpn_cpm_channel]*6
        
        self.latlayer6 = nn.Conv2d(self.fpn_in[5], fpn_channel, kernel_size=1, stride=1, padding=0)
        self.latlayer5 = nn.Conv2d(self.fpn_in[4], fpn_channel, kernel_size=1, stride=1, padding=0)
        self.latlayer4 = nn.Conv2d(self.fpn_in[3], fpn_channel, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(self.fpn_in[2], fpn_channel, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(self.fpn_in[1], fpn_channel, kernel_size=1, stride=1, padding=0)
        self.latlayer1 = nn.Conv2d(self.fpn_in[0], fpn_channel, kernel_size=1, stride=1, padding=0)

        # first is bifpn
        if bifpn:
            self.latlayer6 = BasicConv2d(self.fpn_in[5], fpn_channel, kernel_size=1, stride=1, padding=0)
            self.latlayer5 = BasicConv2d(self.fpn_in[4], fpn_channel, kernel_size=1, stride=1, padding=0)
            self.latlayer4 = BasicConv2d(self.fpn_in[3], fpn_channel, kernel_size=1, stride=1, padding=0)
            self.latlayer3 = BasicConv2d(self.fpn_in[2], fpn_channel, kernel_size=1, stride=1, padding=0)
            self.latlayer2 = BasicConv2d(self.fpn_in[1], fpn_channel, kernel_size=1, stride=1, padding=0)
            self.latlayer1 = BasicConv2d(self.fpn_in[0], fpn_channel, kernel_size=1, stride=1, padding=0)
            
            if not search_fpn and not use_searched_fpn:
                raise NotImplementedError

            elif search_fpn and self.search:
                raise NotImplementedError
                

            elif (search_fpn and not self.search) or use_searched_fpn:
                if search_fpn and not self.search:
                    print("Retraining searched BiFPN...")
                else:
                    print("Using searched BiFPN modules...")
                
                self.bi_fpn = nn.ModuleList(
                    [
                        BiFPN_Neck_From_Genotype(
                            genotype=self.searched_fpn_genotype,
                            feature_size=fpn_channel,
                            weight_node=self.weight_node,
                            fpn_layers=self.fpn_layers
                        ) for _ in range(self.num_layers)
                    ]
                )
                

        elif fpn:
            raise NotImplementedError

        if cpm and not search_cpm:
            raise NotImplementedError

        elif search_cpm and self.search:
            raise NotImplementedError

        elif (search_cpm and not self.search) or use_searched_cpm:
            raise NotImplementedError
    

    def freeze_backbone_BN(self):
        raise NotImplementedError


    def freeze_stem_layer(self):
        raise NotImplementedError

    """
    FPN and CPM modules forward function
    """
    def FPN_CPM_forward(self, sources):
        for index in range(self.num_layers):
            # """ BiFPN """
            if bifpn:
                if (search_fpn and not self.search) or use_searched_fpn:
                    sources = self.bi_fpn[index](sources)
                else:
                    raise NotImplementedError
            
            # """ FPN """
            elif fpn:
                raise NotImplementedError

        return sources

    
    def forward(self, input, epoch=-1):
        self.epoch = epoch
        if isinstance(input, list) or isinstance(input, tuple):
            input, targets = input

        if self.phase == "train":
            raise NotImplementedError
        else:
            priors = None

        # print("None" if targets is None else len(targets), input.size())
        image_size = [input.shape[2], input.shape[3]]
        input = self.stem_layer(input)
        conv1 = self.layer1(input)
        conv2 = self.layer2(conv1)
        conv3 = self.layer3(conv2)
        conv4 = self.layer4(conv3)
        conv5 = self.layer5(conv4)
        conv6 = self.layer6(conv5)
        # sources = [conv1, conv2, conv3, conv4, conv5, conv6]

        sources = [
            self.latlayer1(conv1),
            self.latlayer2(conv2),
            self.latlayer3(conv3),
            self.latlayer4(conv4),
            self.latlayer5(conv5),
            self.latlayer6(conv6),
        ]
        
        \
        sources_arm = None

        sources = self.FPN_CPM_forward(sources)
        
        output = self.head_loc_conf_centerness(sources, image_size, priors=priors, source_arm=sources_arm, arm_data=None if (not auxiliary_refine or self.training) else auxiliary_output)
        
        return output


    def head_loc_conf_centerness(self, source, image_size, priors=None, source_arm=None, arm_data=None):
        # print(arm_data)
        if refine:
            arm_loc = []
            arm_conf = []
            for x, h_loc, h_conf in zip(source_arm, self.arm_loc_head, self.arm_conf_head):
                arm_loc.append(
                    h_loc(x).permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, 4)
                )
                arm_conf.append(
                    h_conf(x).permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, 2)
                )
            arm_loc = torch.cat(arm_loc, dim=1)
            arm_conf = torch.cat(arm_conf, dim=1)
        
        loc = []
        conf = []

        featuremap_size = []
        for level, (x, b_loc, b_conf, h_loc, h_conf) in enumerate(
            zip(source, self.loc_layers, self.conf_layers, self.loc_head, self.conf_head)
        ):
            featuremap_size.append([x.shape[2], x.shape[3]])
            # features after regression and classification branches
            f_loc = b_loc(x)
            f_conf = b_conf(x)

            loc.append(h_loc(f_loc).permute(0, 2, 3, 1).contiguous())
            
            if mo:
                if len(conf) == 0:
                    chunk = torch.chunk(h_conf(f_conf), 4, 1)
                    bmax = torch.max(torch.max(chunk[0], chunk[1]), chunk[2])
                    cls1 = torch.cat([bmax, chunk[3]], dim=1)
                    conf.append(cls1.permute(0, 2, 3, 1).contiguous())
                else:
                    conf.append(h_conf(f_conf).permute(0, 2, 3, 1).contiguous())
            elif imio:
                if cfg["mbox"][0] == 1:
                    cls = self.improved_mio_module(h_conf(f_conf))
                else:
                    raise ValueError("it is cannot be supported for cfg[\"mbox\"][0] > 1")
                conf.append(cls.permute(0, 2, 3, 1).contiguous())
            elif mio:
                len_conf = len(conf)
                if cfg["mbox"][0] == 1:
                    cls = self.mio_module(h_conf(f_conf), len_conf)
                else:
                    mmbox = torch.chunk(h_conf(f_conf), cfg["mbox"][0], 1)
                    cls_0 = self.mio_module(mmbox[0], len_conf)
                    cls_1 = self.mio_module(mmbox[1], len_conf)
                    cls_2 = self.mio_module(mmbox[2], len_conf)
                    cls_3 = self.mio_module(mmbox[3], len_conf)
                    cls = torch.cat([cls_0, cls_1, cls_2, cls_3], dim=1)
                conf.append(cls.permute(0, 2, 3, 1).contiguous())
            else:
                conf.append(h_conf(f_conf).permute(0, 2, 3, 1).contiguous())

        if pa:
            mbox_num = cfg["mbox"][0]
            face_loc = [o[..., : 4 * mbox_num].contiguous().view(o.size(0), -1, 4) for o in loc]
            face_conf = [o[..., : 2 * mbox_num].contiguous().view(o.size(0), -1, self.num_classes) for o in conf]
            head_loc = [o[..., 4 * mbox_num : 8 * mbox_num].contiguous().view(o.size(0), -1, 4) for o in loc[1:]]
            head_conf = [
                o[..., 2 * mbox_num : 4 * mbox_num].contiguous().view(o.size(0), -1, self.num_classes) for o in conf[1:]
            ]
            body_loc = [o[..., 8 * mbox_num :].contiguous().view(o.size(0), -1, 4) for o in loc[2:]]
            body_conf = [o[..., 4 * mbox_num :].contiguous().view(o.size(0), -1, self.num_classes) for o in conf[2:]]
        else:
            face_loc = [o.view(o.size(0), -1, 4) for o in loc]
            face_conf = [o.view(o.size(0), -1, 2) for o in conf]

        batch = face_loc[0].shape[0]

        if priors is None:
            self.cfg["feature_maps"] = featuremap_size
            self.cfg["min_dim"] = image_size
            priors = self.init_priors(self.cfg)

        # adjust the shape of priors
        if isinstance(priors, list) or isinstance(priors, tuple):
            face_priors, head_priors, body_priors = priors
            if face_priors.shape[0] != batch:
                face_priors = (
                    face_priors.view([1] + list(face_priors.shape)).expand([batch] + list(face_priors.shape)).cuda()
                )
                head_priors = (
                    head_priors.view([1] + list(head_priors.shape)).expand([batch] + list(head_priors.shape)).cuda()
                )
                body_priors = (
                    body_priors.view([1] + list(body_priors.shape)).expand([batch] + list(body_priors.shape)).cuda()
                )
        else:
            face_priors = priors
            if face_priors.shape[0] != batch:
                face_priors = (
                    face_priors.view([1] + list(face_priors.shape)).expand([batch] + list(face_priors.shape)).cuda()
                )

        # 
        num_classes = self.num_classes-1 if use_sigmoid else self.num_classes
        
        # return
        # output
        if self.phase == "test":
            face_conf = self.softmax(
                torch.cat(face_conf, dim=1).view(batch, -1, num_classes)
            )[..., 1:]
           
            conf_p = face_conf

            output = self.detect(
                torch.cat(face_loc, dim=1).view(batch, -1, 4),  # loc preds
                conf_p,  # conf preds
                face_priors.type_as(x),  # default boxes
            )
                
            return output

        else:
            raise NotImplementedError
        
    def init_priors(self, cfg, min_size=cfg["min_sizes"], max_size=cfg["max_sizes"], batch=1):
        priorbox = PriorBox(cfg, min_size, max_size, batch=batch)
        with torch.no_grad():
            prior = priorbox.forward().cuda()
        return prior

    def _initialize_alphas(self):
        raise NotImplementedError

    # if not search, using softmax; otherwise, using gumbel_trick
    def alpha_normal(self, INT=True, search=False):
        raise NotImplementedError

    def alpha_fpn(self, INT=True):
        raise NotImplementedError

    def arch_parameters(self):
        return self._arch_parameters

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True) + y

    def _upsample_product(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True) * y

    def mio_module(self, each_mmbox, len_conf):
        chunk = torch.chunk(each_mmbox, each_mmbox.shape[1], 1)
        bmax = torch.max(torch.max(chunk[0], chunk[1]), chunk[2])
        cls = torch.cat([bmax, chunk[3]], dim=1) if len_conf == 0 else torch.cat([chunk[3], bmax], dim=1)
        if len(chunk) == 6:
            cls = torch.cat([cls, chunk[4], chunk[5]], dim=1)
        elif len(chunk) == 8:
            cls = torch.cat([cls, chunk[4], chunk[5], chunk[6], chunk[7]], dim=1)
        return cls

    def improved_mio_module(self, each_mmbox):
        chunk = torch.chunk(each_mmbox, each_mmbox.shape[1], dim=1)
        bmax = torch.max(torch.max(chunk[0], chunk[1]), chunk[2])
        fmax = torch.max(torch.max(chunk[3], chunk[4]), chunk[5])
        cls = torch.cat([bmax, fmax], dim=1)
        if len(chunk) > 6:
            cls = torch.cat([cls] + list(chunk[6:]), dim=1)
        return cls

    def multibox(self, output_channels, mbox_cfg, num_classes):
        # assert margin_loss in ["", "arcface", "cosface", "arcface_scale", "cosface_scale"]
        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()

        for channel in output_channels:
            self.loc_layers.append(DeepHead(channel, stack_convs=stack_convs))
            self.conf_layers.append(DeepHead(channel, stack_convs=stack_convs))

        self.loc_head = nn.ModuleList()
        self.conf_head = nn.ModuleList()

        for k in range(len(output_channels)):
            if pa:
                if k == 0:
                    loc_output = 4
                    conf_output = 2
                elif k == 1:
                    loc_output = 8
                    conf_output = 4
                else:
                    loc_output = 12
                    conf_output = 6
            else:
                loc_output = 4
                conf_output = 2
            
            if use_sigmoid:
                conf_output //= 2

            self.loc_head.append(nn.Conv2d(fpn_cpm_channel, mbox_cfg[k] * loc_output, 1, 1, 0))

            
            if imio:
                print("Improved mio module is employed.")
                self.conf_head.append(nn.Conv2d(fpn_cpm_channel, mbox_cfg[k] * (4 + conf_output), 1, 1, 0))
            elif mio:
                self.conf_head.append(nn.Conv2d(fpn_cpm_channel, mbox_cfg[k] * (2 + conf_output), 1, 1, 0))
            else:
                self.conf_head.append(nn.Conv2d(fpn_cpm_channel, mbox_cfg[k] * conf_output, 1, 1, 0))
            
            

class DeepHead(nn.Module):
    """want to replace the original DeepHeadModule class, by aalenzhang"""

    def __init__(self, input_channels, stack_convs=3):
        super(DeepHead, self).__init__()

        # output_channels = min(input_channels, 256)
        output_channels = input_channels
        self.layers = nn.ModuleList()
        for i in range(stack_convs):
            if i == 0:
                self.layers.append(nn.Conv2d(input_channels, output_channels, 3, 1, 1))
            else:
                # self.layers.append(nn.Conv2d(output_channels, output_channels, 3, 1, 1))
                self.layers.append(
                    nn.Sequential(
                        nn.Conv2d(output_channels, output_channels, 3, 1, 1, groups=output_channels),
                        nn.Conv2d(output_channels, output_channels, 1, 1, 0)
                    )
                )

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x

