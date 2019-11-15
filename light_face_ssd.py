from __future__ import division , print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import  widerface_640
import os
import pdb
import torchvision
import time
from layers import *
cfg = widerface_640

mo = cfg['max_out']
fpn = cfg['feature_pyramid_network']
cpm = cfg['context_predict_module']
mio = cfg['max_in_out']
pa = cfg['pyramid_anchor']
backbone = cfg['backbone']
bup = cfg['bottom_up_path']
refine = cfg['refinedet']

assert(not mo or not mio)

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class Inception2d(nn.Module):

  def __init__(self, in_channels , out_channels=None):
    super(Inception2d, self).__init__()
    mid_channels = int(in_channels/8)
    out_channels = int(in_channels/4)
    self.branch1x1 = BasicConv2d(in_channels, out_channels, kernel_size=1, padding=0)
    self.branch1x1_2 = BasicConv2d(in_channels, out_channels, kernel_size=1, padding=0)
    self.branch3x3_reduce = BasicConv2d(in_channels, mid_channels, kernel_size=1, padding=0)
    self.branch3x3 = BasicConv2d(mid_channels, out_channels, kernel_size=3, padding=1)
    self.branch3x3_reduce_2 = BasicConv2d(in_channels, mid_channels, kernel_size=1, padding=0)
    self.branch3x3_2 = BasicConv2d(mid_channels, out_channels, kernel_size=3, padding=1)
    self.branch3x3_3 = BasicConv2d(out_channels, out_channels, kernel_size=3, padding=1)

  def forward(self, x):
    branch1x1 = self.branch1x1(x)

    branch1x1_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch1x1_2 = self.branch1x1_2(branch1x1_pool)

    branch3x3_reduce = self.branch3x3_reduce(x)
    branch3x3 = self.branch3x3(branch3x3_reduce)

    branch3x3_reduce_2 = self.branch3x3_reduce_2(x)
    branch3x3_2 = self.branch3x3_2(branch3x3_reduce_2)
    branch3x3_3 = self.branch3x3_3(branch3x3_2)

    outputs = [branch1x1, branch1x1_2, branch3x3, branch3x3_3]
    return torch.cat(outputs, 1)


class CRelu(nn.Module):

  def __init__(self, in_channels, out_channels, **kwargs):
    super(CRelu, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
    self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)

  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = torch.cat([x, -x], 1)
    x = F.relu(x, inplace=True)
    return x

class SSD(nn.Module):

    def __init__(self, phase, size, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        assert(num_classes == 2)
        self.cfg = cfg
        self.size = size
        
        if backbone in ['facebox']:

            self.conv1 = CRelu(3, 32, kernel_size=7, stride=4, padding=3)
            #self.conv2 = BasicConv2d(64, 64, kernel_size=3, stride=2, padding=1)
            self.conv3 = CRelu(64, 64, kernel_size=5, stride=2, padding=2)
            #self.conv4 = BasicConv2d(128, 128, kernel_size=3, stride=2, padding=1)

            self.inception1 = Inception2d(64)
            self.inception2 = Inception2d(64)
            self.inception3 = Inception2d(128)
            self.inception4 = Inception2d(128)

            self.conv5_1 = BasicConv2d(128, 128, kernel_size=1, stride=1, padding=0)
            self.conv5_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)
            self.conv6_1 = BasicConv2d(256, 128, kernel_size=1, stride=1, padding=0)
            self.conv6_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)
            fpn_in = [64, 64, 128, 128, 256, 256]  
            cpm_in = [64, 64, 64, 64, 64, 64]  
            fpn_channel = 64  
            cpm_channels = 64  
            output_channels = cpm_in  

        elif backbone in ['mobilenet']:
            self.base = nn.ModuleList( MobileNet() )
            self.layer1 = nn.Sequential( *[self.base[i] for i in range(0,4)] )
            self.layer2 = nn.Sequential( *[self.base[i] for i in range(4,6)] )
            self.layer3 = nn.Sequential( *[self.base[i] for i in range(6,12)] )
            self.layer4 = nn.Sequential( *[self.base[i] for i in range(12,14)] )
            self.layer5 = nn.Sequential( *[BasicConv(1024, 256, kernel_size=1, stride=1),
                          BasicConv(256, 512, kernel_size=3, stride=2, padding=1)]
            )
            self.layer6 = nn.Sequential( *[BasicConv(512, 128, kernel_size=1, stride=1),
                          BasicConv(128, 256, kernel_size=3, stride=2, padding=1)]
            )
            fpn_in = [128, 256, 512, 1024, 512, 256]
            cpm_in = [128, 128, 128, 128, 128, 128] 
            output_channels = [128, 128, 128, 128, 128, 128] 
            fpn_channel = 128
            cpm_channels = 128

        elif backbone in ['resnet18']:
            resnet = torchvision.models.resnet18(pretrained=True)
            self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
            self.layer2 = nn.Sequential(resnet.layer2)
            self.layer3 = nn.Sequential(resnet.layer3)
            self.layer4 = nn.Sequential(resnet.layer4)
            self.layer5 = nn.Sequential( *[BasicConv(512, 128, kernel_size=1, stride=1),
                          BasicConv(128, 256, kernel_size=3, stride=2, padding=1)]
             )
            self.layer6 = nn.Sequential( *[BasicConv(256, 64, kernel_size=1, stride=1),
                          BasicConv(64, 128, kernel_size=3, stride=2, padding=1)]
             )
            fpn_in = [64, 128, 256, 512, 256, 128]
            cpm_in = [128, 128, 128, 128, 128, 128] 
            output_channels = [64, 128, 256, 512, 256, 128]
            fpn_channel = 128
            cpm_channels = 128

        #output_channels = [64, 64, 128, 128, 256, 256]
        #output_channels = [64, 128, 256, 512, 256, 128]
        if fpn:    
            self.smooth3 = nn.Conv2d( fpn_channel, fpn_channel, kernel_size=1, stride=1, padding=0)
            self.smooth2 = nn.Conv2d( fpn_channel, fpn_channel, kernel_size=1, stride=1, padding=0)
            self.smooth1 = nn.Conv2d( fpn_channel, fpn_channel, kernel_size=1, stride=1, padding=0)

            self.latlayer6 = nn.Conv2d( fpn_in[5], fpn_channel, kernel_size=1, stride=1, padding=0)
            self.latlayer5 = nn.Conv2d( fpn_in[4], fpn_channel, kernel_size=1, stride=1, padding=0)
            self.latlayer4 = nn.Conv2d( fpn_in[3], fpn_channel, kernel_size=1, stride=1, padding=0)
            self.latlayer3 = nn.Conv2d( fpn_in[2], fpn_channel, kernel_size=1, stride=1, padding=0)
            self.latlayer2 = nn.Conv2d( fpn_in[1], fpn_channel, kernel_size=1, stride=1, padding=0)
            self.latlayer1 = nn.Conv2d( fpn_in[0], fpn_channel, kernel_size=1, stride=1, padding=0)    

        if cpm:
            self.cpm1 = Inception2d(cpm_in[0])
            self.cpm2 = Inception2d(cpm_in[1])
            self.cpm3 = Inception2d(cpm_in[2])
            self.cpm4 = Inception2d(cpm_in[3])
            self.cpm5 = Inception2d(cpm_in[4])
            self.cpm6 = Inception2d(cpm_in[5])

        if pa:
            face_head = face_multibox(output_channels, cfg['mbox'], num_classes , cpm_channels)  
            self.loc = nn.ModuleList(face_head[0])
            self.conf = nn.ModuleList(face_head[1])
            if phase == 'train':
              pa_head = pa_multibox(output_channels, cfg['mbox'], num_classes , cpm_channels)  
              self.pa_loc = nn.ModuleList(pa_head[0])
              self.pa_conf = nn.ModuleList(pa_head[1])
        else:
            head = multibox(output_channels, cfg['mbox'], num_classes)  
            self.loc = nn.ModuleList(head[0])
            self.conf = nn.ModuleList(head[1])
         
        if refine:
            arm_head = arm_multibox(output_channels , cfg['mbox'], num_classes)
            self.arm_loc = nn.ModuleList(arm_head[0])
            self.arm_conf = nn.ModuleList(arm_head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, cfg['num_thresh'], cfg['conf_thresh'], cfg['nms_thresh'])
        
        if phase == 'train':
          print ("init weight!")
          for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_() 
        
    def init_priors(self ,cfg , min_size=cfg['min_sizes'], max_size=cfg['max_sizes']):
        priorbox = PriorBox(cfg , min_size, max_size)
        prior = Variable ( priorbox.forward() , volatile=True)
        return prior
        
    def forward(self, x):

        t0 = time.time()

        image_size = [x.shape[2] , x.shape[3]]
        loc = list()
        conf = list()

        if backbone in ['facebox']:
            conv1_x = self.inception1( self.conv1(x) )
            conv2_x = self.inception2( F.max_pool2d(conv1_x, kernel_size=3, stride=2, padding=1) )
            conv3_x = self.inception3( self.conv3(conv2_x) )
            conv4_x = self.inception4( F.max_pool2d(conv3_x, kernel_size=3, stride=2, padding=1) )
            conv5_x = self.conv5_2(self.conv5_1(conv4_x))
            conv6_x = self.conv6_2(self.conv6_1(conv5_x))

        elif backbone in ['mobilenet']:
            conv1_x = self.layer1(x)
            conv2_x = self.layer2(conv1_x)
            conv3_x = self.layer3(conv2_x)
            conv4_x = self.layer4(conv3_x)
            conv5_x = self.layer5(conv4_x)
            conv6_x = self.layer6(conv5_x)        
        elif backbone in ['resnet18']:
            conv1_x = self.layer1(x)
            conv2_x = self.layer2(conv1_x)
            conv3_x = self.layer3(conv2_x)
            conv4_x = self.layer4(conv3_x)
            conv5_x = self.layer5(conv4_x)
            conv6_x = self.layer6(conv5_x)
        t1 = time.time()

        if refine:   
            arm_loc = list()
            arm_conf = list()
            arm_sources = [conv3_3_x, conv4_3_x, conv5_3_x, fc7_x, conv6_2_x, conv7_2_x]
            for (x, l, c) in zip(arm_sources, self.arm_loc, self.arm_conf):
                arm_loc.append( l(x).permute(0, 2, 3, 1).contiguous() )    
                arm_conf.append( c(x).permute(0, 2, 3, 1).contiguous() )
            arm_loc = torch.cat([o.view(o.size(0), -1) for o in arm_loc], 1)
            arm_conf = torch.cat([o.view(o.size(0), -1) for o in arm_conf], 1)
              
        if fpn:
            conv6_x = self.latlayer6(conv6_x)
            conv5_x = self.latlayer5(conv5_x)
            conv4_x = self.latlayer4(conv4_x)
            conv3_x = self.latlayer3(conv3_x)
            conv2_x = self.latlayer2(conv2_x)
            conv1_x = self.latlayer1(conv1_x)
    
            conv3_x = self.smooth3( self._upsample_product( conv4_x , conv3_x ) )
            conv2_x = self.smooth2( self._upsample_product( conv3_x , conv2_x ) )
            conv1_x = self.smooth1( self._upsample_product( conv2_x , conv1_x ) )

        t2 = time.time()
              
        sources = [conv1_x, conv2_x, conv3_x, conv4_x, conv5_x, conv6_x]
        if cpm:
           sources[0] = self.cpm1(sources[0])
           sources[1] = self.cpm2(sources[1])
           sources[2] = self.cpm3(sources[2])
           sources[3] = self.cpm4(sources[3])
           sources[4] = self.cpm5(sources[4])
           sources[5] = self.cpm6(sources[5])

        t3 = time.time()        

        # apply multibox head to source layers
        featuremap_size = []
        for  (x, l, c) in zip(sources, self.loc, self.conf):
            featuremap_size.append( [x.shape[2], x.shape[3]] )
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        t4 = time.time()

        if pa and self.phase == "train":
            pa_loc = list()
            pa_conf = list()
            for (x, l, c) in zip(sources[1:], self.pa_loc, self.pa_conf):
              pa_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
              pa_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            mbox_num = cfg['mbox'][0]
            head_loc = torch.cat( [o[:,:,:,:4*mbox_num].contiguous().view(o.size(0),-1) for o in pa_loc],1)
            head_conf = torch.cat( [o[:,:,:,:2*mbox_num].contiguous().view(o.size(0),-1) for o in pa_conf],1)           
            body_loc = torch.cat( [o[:,:,:,4*mbox_num:8*mbox_num].contiguous().view(o.size(0),-1) for o in pa_loc[1:]],1)
            body_conf = torch.cat( [o[:,:,:,2*mbox_num:4*mbox_num].contiguous().view(o.size(0),-1) for o in pa_conf[1:]],1)

        face_loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        face_conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        t5= time.time()

        if self.phase == "test":
            self.cfg['feature_maps'] = featuremap_size
            self.cfg['min_dim'] = image_size
            self.priors = self.init_priors(self.cfg)
            if refine:
                output = self.detect(
                  face_loc.view(face_loc.size(0), -1, 4),         # loc preds
                  self.softmax(face_conf.view(face_conf.size(0), -1, self.num_classes)), # conf preds
                  self.priors.type(type(x.data)),                  # default boxes
                  arm_loc.view(arm_loc.size(0), -1, 4),
                  self.softmax(arm_conf.view(arm_conf.size(0), -1, self.num_classes)),
                )
            else:
                output = self.detect(
                  face_loc.view(face_loc.size(0), -1, 4),         # loc preds
                  self.softmax(face_conf.view(face_conf.size(0), -1, self.num_classes)), # conf preds
                  self.priors.type(type(x.data))                 # default boxes
                )
            t6 = time.time()

        else:
            self.cfg['feature_maps'] = featuremap_size
            self.cfg['min_dim'] = image_size
            if pa: 
              self.face_priors = self.init_priors(self.cfg)
              self.head_priors = self.init_priors(self.cfg , min_size=cfg['min_sizes'][:-1], max_size=cfg['max_sizes'][:-1])
              self.body_priors = self.init_priors(self.cfg , min_size=cfg['min_sizes'][:-2], max_size=cfg['max_sizes'][:-2])
              output = (
                face_loc.view(face_loc.size(0), -1, 4),
                face_conf.view(face_conf.size(0), -1, self.num_classes),
                self.face_priors,
 
                head_loc.view(head_loc.size(0), -1, 4),
                head_conf.view(head_conf.size(0), -1, self.num_classes),
                self.head_priors,

                body_loc.view(body_loc.size(0), -1, 4),
                body_conf.view(body_conf.size(0), -1, self.num_classes),
                self.body_priors
              )
            else:
              self.priors = self.init_priors(self.cfg)
              output = (
                face_loc.view(face_loc.size(0), -1, 4),
                face_conf.view(face_conf.size(0), -1, self.num_classes),
                self.priors
              )
            if refine:
                output = output + tuple((arm_loc.view(arm_loc.size(0), -1, 4), arm_conf.view(arm_conf.size(0), -1, self.num_classes) ))

        if self.phase == "test":
          print( 'Backbone: %.4f , FPN: %.4f, CPM: %.4f, Head: %.4f, PA: %.4f, Decode_NMS: %.4f' % (t1-t0 , t2-t1, t3-t2, t4-t3, t5-t4, t6-t5))
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def _upsample_product(self, x, y):
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') * y

def multibox(input_channels, mbox_cfg, num_classes):
    loc_layers = []
    conf_layers = []
    for k, v in enumerate(input_channels):
        loc_layers  += [nn.Conv2d(input_channels[k], mbox_cfg[k] * 4, kernel_size=3, dilation=1, stride=1, padding=1)]
        conf_layers += [nn.Conv2d(input_channels[k], mbox_cfg[k] * num_classes, kernel_size=3, dilation=1, stride=1, padding=1)]
    return loc_layers, conf_layers

def arm_multibox(output_channels, mbox_cfg, num_classes):
    loc_layers = []
    conf_layers = []
    for k, v in enumerate(output_channels):
        input_channels =  v
        loc_layers += [nn.Conv2d(input_channels, mbox_cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(input_channels,  mbox_cfg[k] * num_classes, kernel_size=3, padding=1)]
    return loc_layers, conf_layers

class DeepHeadModule(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DeepHeadModule , self).__init__()
        self._input_channels = input_channels
        self._mid_channels = 16
        self._output_channels = output_channels
        self.conv1 = BasicConv2d(self._input_channels, self._mid_channels, kernel_size=1, dilation=1, stride=1, padding=0)
        self.conv2 = BasicConv2d(self._mid_channels, self._mid_channels, kernel_size=3, dilation=1, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self._mid_channels, self._output_channels, kernel_size=1, dilation=1, stride=1, padding=0)
    def forward(self, x):
        #return self.conv3( F.relu(self.conv2( F.relu(self.conv1(x), inplace=True) )inplace=True) )
        return self.conv3(self.conv2(self.conv1(x)))

def face_multibox(output_channels, mbox_cfg, num_classes , cpm_c):
    loc_layers = []
    conf_layers = []
    for k, v in enumerate(output_channels):
        input_channels = (cpm_c if cpm else v)
        loc_output = 4
        conf_output = 2
        loc_layers  += [ DeepHeadModule(input_channels, mbox_cfg[k] * loc_output) ]
        conf_layers += [ DeepHeadModule(input_channels, mbox_cfg[k] * conf_output)]
    return (loc_layers, conf_layers)

def pa_multibox(output_channels, mbox_cfg, num_classes , cpm_c):
    loc_layers = []
    conf_layers = []
    for k, v in enumerate(output_channels):
        input_channels = (cpm_c if cpm else v)
        if k == 0:
           continue
        elif k == 1:
            loc_output = 4
            conf_output = 2
        else:
            loc_output = 8
            conf_output = 4
        loc_layers += [ DeepHeadModule(input_channels, mbox_cfg[k] * loc_output) ]
        conf_layers += [DeepHeadModule(input_channels, mbox_cfg[k] * conf_output)]
    return (loc_layers, conf_layers)


def build_ssd(phase, size=640, num_classes=2):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size!=640:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD640 (size=640) is supported!")
    return SSD(phase, size, num_classes)
