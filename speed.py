from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import WIDERFace_ROOT , WIDERFace_CLASSES as labelmap
from PIL import Image
from data import WIDERFaceDetection, WIDERFaceAnnotationTransform, WIDERFace_CLASSES, WIDERFace_ROOT, BaseTransform , TestBaseTransform
from data import *
import torch.utils.data as data
from light_face_ssd import build_ssd
import pdb
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import time
from layers.box_utils import nms
import torchvision
#from torchscope import scope

plt.switch_backend('agg')

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/light_DSFD.pth',
                    type=str, help='Trained state_dict file path to open')
#parser.add_argument('--save_folder', default='eval_tools/WIDERFace_SSD_F_RES152_FPN_MIO_CPM_PA0.70.3_DAS_Anchor1.5_IOU0.4_DeepClsRelu_6W_16bs_pyramidtest_testset/', type=str,
#                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.01, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--widerface_root', default=WIDERFace_ROOT, help='Location of VOC root directory')
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
#if not os.path.exists(args.save_folder):
#    os.mkdir(args.save_folder)

def detect_face(image, shrink):
    x = image
    if shrink != 1:
        x = cv2.resize(image, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR)
    print('shrink:{}'.format(shrink))
    width = x.shape[1]
    height = x.shape[0]
    print (width , height)
    x = x.astype(np.float32)
    x -= np.array([104, 117, 123],dtype=np.float32)

    x = torch.from_numpy(x).permute(2, 0, 1)
    x = x.unsqueeze(0)
    if args.cuda:
        x = Variable(x.cuda(), volatile=True)
    else:
        #with torch.no_grad():
        #    x = Variable(x)
        x = Variable(x, volatile=True)
    
    #net.priorbox = PriorBoxLayer(width,height)
    #model = torchvision.models.resnet18(pretrained=True)
    time0 = time.time()
    #model(x)
    #time01 = time.time()
    #print ( time01 - time0)
    y = net(x)
    time1 = time.time()
    print ( "infer time: %.4f " % (time1-time0) )
    detections = y.data
    scale = torch.Tensor([width, height, width, height])

    boxes=[]
    scores = []
    for i in range(detections.size(1)):
        j = 0
        while detections[0,i,j,0] >= 0.01:
            score = detections[0,i,j,0]
            pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
            boxes.append([pt[0],pt[1],pt[2],pt[3]])
            scores.append(score)
            j += 1
            if j >= detections.size(2):
                break

    det_conf = np.array(scores)
    boxes = np.array(boxes)

    if boxes.shape[0] == 0:
        return np.array([[0,0,0,0,0.001]])

    det_xmin = boxes[:,0] / shrink
    det_ymin = boxes[:,1] / shrink
    det_xmax = boxes[:,2] / shrink
    det_ymax = boxes[:,3] / shrink
    det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))

    keep_index = np.where(det[:, 4] >= 0)[0]
    det = det[keep_index, :]
    return det

# load net
cfg = widerface_640
num_classes = len(WIDERFace_CLASSES) + 1 # +1 background
net = build_ssd('test', cfg['min_dim'], num_classes) # initialize SSD
#net = nn.DataParallel(net)
net.load_state_dict(torch.load(args.trained_model))
if args.cuda:
    net.cuda()
net.eval()
print('Finished loading model!')

#scope( net, input_size=( 3, 720, 1280) , batch_size=1 , device='cpu')

path = './data/yuebing.jpg'
image = cv2.imread(path, cv2.IMREAD_COLOR)
image = cv2.resize(image, (720, 1280), interpolation=cv2.INTER_LINEAR)
#(720,1280)
shrink=1
time1 = time.time()
for i in range(100):
  det = detect_face(image, shrink)  # origin test
time2 = time.time()
print( (time2-time1)/100 )




