'''
@Author: aalenzhang
@Date: 2020-01-28 19:46:14
@LastEditors: Please set LastEditors
@LastEditTime: 2020-03-06 11:28:29
@Description: 
@FilePath: \DSFDv2_r18\layers\functions\detection.py
'''
from __future__ import division
import torch
import torch.nn as nn
from ..box_utils import decode, nms, center_size
from dataset import widerface_640 as cfg
import pdb


class Detect(nn.Module):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        super(Detect, self).__init__()
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']

    def forward(self, loc_data, conf_data, prior_data, arm_loc_data=None, arm_conf_data=None):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)  # batch size
        prior_data = prior_data[0]
        num_priors = prior_data.size(0)

        # swordli
        # num_priors = loc_data.size(1)

        output = torch.zeros(num, conf_data.shape[-1], self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors, -1).transpose(2, 1)
        if cfg['refinedet']:
            conf_preds_arm = arm_conf_data.view(num, num_priors, -1).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            default = prior_data
    
            filtered_mask = torch.ones_like(conf_preds[i]) >= 1
            if arm_conf_data is not None:
                # print(arm_conf_data.min(), arm_conf_data.max(), arm_conf_data.shape)
                filtered_mask[..., :-8525] = arm_conf_data[..., :-8525] >= 0.01
            
            decoded_boxes = decode(loc_data[i], default, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()

            for cl in range(conf_scores.shape[0]):
                c_mask = conf_scores[cl].gt(self.conf_thresh) & filtered_mask[cl]
                scores = conf_scores[cl][c_mask]
                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                # ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                # output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)
                
                if scores.shape[-1] <= self.top_k:
                    count = scores.shape[-1]
                else:
                    count = self.top_k
                    _, idx = torch.topk(scores, k=self.top_k, largest=True)
                    boxes = boxes[idx]
                    scores = scores[idx]
                output[i, cl, :count] = torch.cat([scores.unsqueeze(1), boxes], dim=1)
                
        return output
