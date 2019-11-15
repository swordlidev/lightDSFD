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
#from resnet50_ssd import build_sfd
import pdb
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import time
plt.switch_backend('agg')

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/light_DSFD.pth',
                    type=str, help='Trained state_dict file path to open')

parser.add_argument('--save_folder', default='eval_tools/light_DSFD/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.01, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--widerface_root', default=WIDERFace_ROOT, help='Location of VOC root directory')
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)
        # get needed merge det and delete these det
        merge_index = np.where(o >= 0.3)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)
        if merge_index.shape[0] <= 1:
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum
    dets = dets[0:750, :]
    return dets

def write_to_txt(f, det , event , im_name):
    f.write('{:s}\n'.format(event + '/' + im_name))
    f.write('{:d}\n'.format(det.shape[0]))
    for i in range(det.shape[0]):
        xmin = det[i][0]
        ymin = det[i][1]
        xmax = det[i][2]
        ymax = det[i][3]
        score = det[i][4] 
        f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.
                format(xmin, ymin, (xmax - xmin + 1), (ymax - ymin + 1), score))

def infer(net , img , transform , thresh , cuda , shrink):
    if shrink != 1:
        img = cv2.resize(img, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR)

    x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0) , volatile=True)
    if cuda:
        x = x.cuda()
    y = net(x)      # forward pass
    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor([ img.shape[1]/shrink, img.shape[0]/shrink,
                         img.shape[1]/shrink, img.shape[0]/shrink] )
    det = []
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= thresh:
            score = detections[0, i, j, 0]
            #label_name = labelmap[i-1]
            pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
            coords = (pt[0], pt[1], pt[2], pt[3]) 
            det.append([pt[0], pt[1], pt[2], pt[3], score])
            j += 1
    if (len(det)) == 0:
        det = [ [0.1,0.1,0.2,0.2,0.01] ]
    det = np.array(det)

    keep_index = np.where(det[:, 4] >= 0)[0]
    det = det[keep_index, :]
    return det

def vis_detections(im,  dets, image_name , thresh=0.5):
    """Draw detected bounding boxes."""
    class_name = 'face'
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    print (len(inds))
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=2.5)
            )
        '''
        ax.text(bbox[0], bbox[1] - 5,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=10, color='white')
        '''
    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('./data/'+image_name, dpi=fig.dpi)


def light_test_oneimage():
    # load net
    cfg = widerface_640
    num_classes = len(WIDERFace_CLASSES) + 1 # +1 background
    net = build_ssd('test', cfg['min_dim'], num_classes) # initialize SSD
    #net = nn.DataParallel(net)
    net.load_state_dict(torch.load(args.trained_model))
    net.cuda()
    net.eval()
    print('Finished loading model!')

    # evaluation
    cuda = args.cuda
    transform = TestBaseTransform((104, 117, 123))
    thresh=cfg['conf_thresh']

    # load data
    #path = './data/worlds-largest-selfie-biggest.jpg'
    #path = "./data/worlds-largest-selfie.jpg"
    path = "./data/yuebing.jpg"
    img_id = 'result'
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    shrink = 1
    det = infer(net , img , transform , thresh , cuda , shrink)
    vis_detections(img , det , img_id, 0.6)


def light_test_widerface():
    # load net
    cfg = widerface_640
    num_classes = len(WIDERFace_CLASSES) + 1 # +1 background
    net = build_ssd('test', cfg['min_dim'], num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.cuda()
    net.eval()
    print('Finished loading model!')
    # load data
    testset = WIDERFaceDetection(args.widerface_root, 'val' , None, WIDERFaceAnnotationTransform())
    # evaluation
    cuda = args.cuda
    transform = TestBaseTransform((104, 117, 123))
    thresh=cfg['conf_thresh']
    print (thresh)
    save_path = args.save_folder
    num_images = len(testset)
    for i in range(num_images):
        img = testset.pull_image(i)
        img_id, annotation = testset.pull_anno(i)
        event = testset.pull_event(i)
        print('Testing image {:d}/{:d} {}....'.format(i+1, num_images , img_id))
        shrink = 1
        det = infer(net , img , transform , thresh , cuda , shrink)
        if not os.path.exists(save_path + event):
            os.makedirs(save_path + event)
        f = open(save_path + event + '/' + img_id.split(".")[0] + '.txt', 'w')
        write_to_txt(f, np.array(det) , event , img_id)

if __name__ == '__main__':
    light_test_oneimage()
    #light_test_widerface()
