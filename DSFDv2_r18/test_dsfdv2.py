'''
@Author: aalenzhang
@Date: 2020-02-19 21:03:44
@LastEditors: Please set LastEditors
@LastEditTime: 2020-03-06 11:00:46
@Description: 
@FilePath: \DSFDv2_r18\test_dsfdv2.py
'''
from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from PIL import Image

# from face_ssd import build_ssd
from model_search import Network
from collections import OrderedDict

import numpy as np
import cv2
import math
import glob
import time
from dataset.config import widerface_640 as cfg
from torchvision.ops import nms as torch_nms

from collections import namedtuple

FPN_Genotype = namedtuple("FPN_Genotype", "Inter_Layer Out_Layer")
AutoFPN = FPN_Genotype(
    Inter_Layer=[
        [("sep_conv_3x3", 1), ("conv_1x1", 0)],
        [("sep_conv_3x3", 2), ("sep_conv_3x3", 0), ("conv_1x1", 1)],
        [("sep_conv_3x3", 3), ("sep_conv_3x3", 1), ("conv_1x1", 2)],
        [("sep_conv_3x3", 4), ("sep_conv_3x3", 2), ("conv_1x1", 3)],
        [("sep_conv_3x3", 5), ("sep_conv_3x3", 3), ("conv_1x1", 4)],
        [("sep_conv_3x3", 4), ("conv_1x1", 5)],
    ],
    Out_Layer=[],
)


parser = argparse.ArgumentParser(description="Single Shot MultiBox Detection")
parser.add_argument(
    "--trained_model",
    default="./weights/dsfdv2_r18.pth",
    type=str,
    help="Trained state_dict file path to open",
)
parser.add_argument("--path", default=None, type=str, help="where images are stored")

parser.add_argument(
    "--thresh", default=0.8, type=float, help="Final confidence threshold"
)

args = parser.parse_args()


def test_base_transform(image, mean):
    #x = cv2.resize(image, (size, size)).astype(np.float32)
    x = image.astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x


class TestBaseTransform:
    def __init__(self, mean):
        #self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return test_base_transform(image, self.mean), boxes, labels


args.cuda = True
if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    torch.set_default_tensor_type("torch.FloatTensor")


net = Network(
    C=64,
    criterion=None,
    num_classes=2,
    layers=1,
    phase="test",
    search=False,
    args=args,
    searched_fpn_genotype=AutoFPN,
    searched_cpm_genotype=None,
    fpn_layers=1,
    cpm_layers=1,
    auxiliary_loss=False,
)

state_dict = torch.load(args.trained_model, map_location=lambda storage, loc: storage)
print("Pretrained model loading OK...")
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if "auxiliary" not in k:
        name = k[7:]  # remove module.
        new_state_dict[name] = v
    else:
        print("Auxiliary loss is used when retraining.")

net.load_state_dict(new_state_dict)
net.cuda()
net.eval()
print("Finished loading model!")

transform = TestBaseTransform((104, 117, 123))

def preprocess(img):
    x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
    x = x.unsqueeze(0).cuda()
    return x
    

save_path = args.path + "_res"
os.makedirs(save_path, exist_ok=True)

files = glob.glob(os.path.join(args.path, "*.png")) + glob.glob(os.path.join(args.path, "*.jpg")) + glob.glob(os.path.join(args.path, "*.bmp"))

with torch.no_grad():
    for f in files:
        img_name = f.split("/")[-1]
        img_cv2 = cv2.imread(f)

        # pre-processing
        img = preprocess(img_cv2)
        
        # post-processing
        detections = net(img).view(-1, 5)
        # scale each detection back up to the image
        scale = torch.Tensor(
            [
                img_cv2.shape[1],
                img_cv2.shape[0],
                img_cv2.shape[1],
                img_cv2.shape[0]
            ]
        ).cuda()
        scores = detections[..., 0]
        boxes = detections[..., 1:] * scale
        
        # filter the boxes whose score is smaller than 0.8
        keep_mask = (scores >= args.thresh) & (boxes[..., -1] > 2.0)
        scores = scores[keep_mask]
        boxes = boxes[keep_mask]
        # print(scores.max())

        keep_idx = torch_nms(boxes, scores, iou_threshold=0.4)
        if len(keep_idx) > 0:
            keep_boxes = boxes[keep_idx].cpu().numpy()
            keep_scores = scores[keep_idx].cpu().numpy()

            for box, s in zip(keep_boxes, keep_scores):
                box = np.array(box, np.int32)
                cv2.rectangle(img_cv2, (box[0], box[1]), (box[2], box[3]), color=(0, 0, 255), thickness=2)
                cv2.rectangle(img_cv2, (box[0], box[1]-20), (box[0]+80, box[1]-2), color=(0, 255, 0), thickness=-1) 
                cv2.putText(img_cv2, "{:.2f}".format(s), (box[0], box[1]-2), cv2.FONT_HERSHEY_SIMPLEX, args.thresh, color=(255,255,255), thickness=2)
            
        cv2.imwrite(os.path.join(save_path, img_name), img_cv2)
        print("{} faces are detected in {}.".format(len(keep_idx), img_name))