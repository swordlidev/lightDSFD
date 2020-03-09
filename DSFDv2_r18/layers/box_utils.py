# -*- coding: utf-8 -*-
from __future__ import division
import torch
import math
import pdb
import numpy as np
from torchvision.ops import nms as _nms


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), 1)  # xmin, ymin  # xmax, ymax


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat([(boxes[:, 2:] + boxes[:, :2]) / 2, boxes[:, 2:] - boxes[:, :2]], 1)  # cx, cy  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def jaccard_diou(box_a, box_b):
    # 
    A = box_a.size(0)
    B = box_b.size(0)
    # iou
    int_max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    int_min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((int_max_xy - int_min_xy), min=0)
    # intersect, shape [A, B]
    inter = inter[:, :, 0] * inter[:, :, 1]
    # union
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    # iou
    iou = inter / union

    # outer_box
    out_max_xy = torch.max(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    out_min_xy = torch.min(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    # center of boxes
    ctr_box_a = ((box_a[:, :2] + box_a[:, 2:]) / 2.0).unsqueeze(1).expand(A, B, 2)
    ctr_box_b = ((box_b[:, :2] + box_b[:, 2:]) / 2.0).unsqueeze(0).expand(A, B, 2)

    r_dious = (ctr_box_a - ctr_box_b).pow(2).sum(dim=-1) / (out_max_xy - out_min_xy).pow(2).sum(dim=-1)
    dious = iou - r_dious

    return dious.clamp(min=-1.0, max=1.0)


def refine_match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx, arm_loc):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # decode arm_loc
    decode_arm_loc = decode(arm_loc, priors=priors, variances=variances)
    # jaccard index
    overlaps = jaccard(truths, decode_arm_loc)
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)

    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)

    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    matches = truths[best_truth_idx]  # Shape: [num_priors,4]

    conf = labels[best_truth_idx] + 1  # Shape: [num_priors] , init conf with all 1a
    conf[best_truth_overlap < threshold] = 0  # label as background
    loc = encode(matches, center_size(decode_arm_loc), variances)

    loc_t[idx] = loc  # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


def match_anchors(truths, priors, threshold=[0.4, 0.7]):
    """
    by aalenzhang
    Match each prior box with the ground truth box of the highest jaccard overlap,
    don't encode the bounding box, then return them 
    """
    overlaps = jaccard(truths, priors)

    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)

    conf = torch.ones_like(best_truth_overlap).long()

    conf[best_truth_overlap < threshold[1]] = -1  # ignore
    conf[best_truth_overlap < threshold[0]] = 0  # background

    remove_mask = torch.ones_like(conf)
    keep_idx = _nms(priors, best_truth_overlap, iou_threshold=0.7)[:512]
    remove_mask.index_fill_(0, keep_idx, 0)

    # after NMS, ignore them
    conf[remove_mask] = -1

    return conf


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx, IoU=False, centerness=False, arm_loc=None):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    # print('truth: {}, priors: {}.'.format(truths.device, priors.device))
    # if arm_loc is not None:
        # print(arm_loc.shape)
        # priors = center_size(decode(arm_loc, priors, variances))
        # priors[-8525:, ...] = center_size(decode(arm_loc[-8525:, ...], priors[-8525:, ...], variances))
    
    overlaps = jaccard(truths, point_form(priors))
    # overlaps = jaccard_diou(truths, point_form(priors))
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)

    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)

    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    matches = truths[best_truth_idx]  # Shape: [num_priors,4]

    conf = labels[best_truth_idx] + 1  # Shape: [num_priors] , init conf with all 1a
    
    if centerness:
        in_gt = is_in_box(priors, matches)
        conf[~in_gt] = -1       # ignore the anchors that locate outside the corresponding gt box
    
    if len(threshold) > 1:
        conf[best_truth_overlap < threshold[1]] = -1  # ignore
        conf[best_truth_overlap < threshold[0]] = 0
    else:
        conf[best_truth_overlap < threshold[0]] = 0  # label as background
    
    # print(conf.max(), conf.min())
    
    if not IoU:
        loc = encode(matches, priors, variances)
    else:
        loc = matches

    loc_t[idx] = loc  # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


def is_in_box(ctr_xy, boxes):
    
    ctr_xy = ctr_xy[..., :2]
    boxes = boxes[..., :4]
    
    l = ctr_xy[..., 0] - boxes[..., 0]
    t = ctr_xy[..., 1] - boxes[..., 1]
    r = boxes[..., 2] - ctr_xy[..., 0]
    b = boxes[..., 3] - ctr_xy[..., 1]

    in_box = torch.stack([l, t, r, b], dim=-1).min(dim=-1)[0] > 0.01
    
    # print(ctr_xy[in_box][0, ...], boxes[in_box][0, ...], in_box[in_box][0])
    
    return in_box


def ATSS_match(truths, priors, variances, labels, loc_t, conf_t, idx, k=9, IoU=False):
    """
    match the predictions with ground truths according to the ATSS method
    Args:
        truths: ground truth, shape [N, 4], [xmin, ymin, xmax, ymax]
        variance: [0.1, 0.2] in general
        labels: [N]
    """

    if priors.shape[0] == 34125:
        feature_size = [160 ** 2, 80 ** 2, 40 ** 2, 20 ** 2, 10 ** 2, 5 ** 2]
    elif priors.shape[0] == 8525:
        feature_size = [80 ** 2, 40 ** 2, 20 ** 2, 10 ** 2, 5 ** 2]
    elif priors.shape[0] == 2125:
        feature_size = [40 ** 2, 20 ** 2, 10 ** 2, 5 ** 2]
    elif priors.shape[0] == 87360:
        feature_size = [256 ** 2, 128 ** 2, 64 ** 2, 32 ** 2, 16 ** 2, 8 ** 2]
    elif priors.shape[0] == 21824:
        feature_size = [128 ** 2, 64 ** 2, 32 ** 2, 16 ** 2, 8 ** 2]
    elif priors.shape[0] == 5440:
        feature_size = [64 ** 2, 32 ** 2, 16 ** 2, 8 ** 2]
    else:
        raise ValueError("Unknown Pyramid Size When Spliting!")

    N_gt = truths.shape[0]
    N_prior = priors.shape[0]

    ctr_gt = (truths[:, :2] + truths[:, 2:]) / 2.0
    ctr_prior = priors[:, :2]

    box_gt = truths
    box_prior = point_form(priors)

    # [N_prior, N_gt]
    overlaps = jaccard(box_prior, box_gt)

    # compute the distance between center point of two boxes
    # [N_prior, N_gt]
    distance = ((ctr_gt.view(1, -1, 2) - ctr_prior.view(-1, 1, 2)) ** 2).sum(dim=-1)

    # firstly, get the top-k minimal anchors for each gt bboxs at per pyramid level
    candidate_idxs = []
    start_idx = 0
    for level_size in feature_size:
        end_idx = start_idx + level_size

        distance_per_level = distance[start_idx:end_idx, :]
        _, topk_idx_per_level = torch.topk(distance_per_level, dim=0, k=k, largest=False)
        candidate_idxs.append(topk_idx_per_level + start_idx)

        start_idx = end_idx
    
    # concat together
    candidate_idxs = torch.cat(candidate_idxs, dim=0)
        
    # _, candidate_idxs = torch.topk(distance, dim=0, k=6*k, largest=False)
    
    # secondly, compute the threshold using the mean and standard variance of iou between anchors and ground-truth boxes
    candidate_iou = overlaps[candidate_idxs, torch.arange(N_gt)]
    iou_mean_per_gt = candidate_iou.mean(dim=0)
    iou_std_per_gt = candidate_iou.std(dim=0)
    iou_thresh_per_gt = iou_mean_per_gt + 0.5*iou_std_per_gt
    is_pos = candidate_iou >= iou_thresh_per_gt

    # thirdly, filter the anchor whose center is outside the gt box
    for ng in range(N_gt):
        candidate_idxs[:, ng] += ng * N_prior
    candidate_idxs = candidate_idxs.view(-1)

    e_ctr_prior = ctr_prior.view(1, -1, 2).expand(N_gt, -1, 2).contiguous().view(-1, 2)
    candidate_ctr_x = e_ctr_prior[candidate_idxs, 0]
    candidate_ctr_y = e_ctr_prior[candidate_idxs, 1]

    l = candidate_ctr_x.view(-1, N_gt) - box_gt[:, 0]
    t = candidate_ctr_y.view(-1, N_gt) - box_gt[:, 1]
    r = box_gt[:, 2] - candidate_ctr_x.view(-1, N_gt)
    b = box_gt[:, 3] - candidate_ctr_y.view(-1, N_gt)

    is_in_gts = torch.stack([l, t, r, b], dim=-1).min(dim=-1)[0] > 0.01
    is_pos = is_pos & is_in_gts

    # if an anchor box is assigned to multiple gts, the one with the highest IoU will be selected.
    # [N_prior, N_gt] -> [N_gt, N_prior] -> [N_gt*N_prior]
    ious_inf = torch.full_like(overlaps, -100).t().contiguous().view(-1)
    index = candidate_idxs.view(-1)[is_pos.view(-1)]
    ious_inf[index] = overlaps.t().contiguous().view(-1)[index]
    ious_inf = ious_inf.view(N_gt, -1).t()

    anchor_to_gt_values, anchor_to_gt_indexs = ious_inf.max(dim=1)
    cls_labels = labels[anchor_to_gt_indexs] + 1  # positive
    cls_labels[anchor_to_gt_values == -100] = 0  # negtive
    matched_gts = truths[anchor_to_gt_indexs]

    # print("pos_num: ", cls_labels.sum().item(), ", total_num: ", N_prior)
    
    if not IoU:
        loc = encode(matched_gts, priors, variances)
    else:
        loc = matched_gts

    loc_t[idx] = loc  # [num_priors,4] encoded offsets to learn
    conf_t[idx] = cls_labels  # [num_priors] top class label for each prior


def pa_sfd_match(part, threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    '''
    S3FD: Single Shot Scale-invariant Face Detector
    '''
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    overlaps = jaccard(truths, point_form(priors))
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)

    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    conf = labels[best_truth_idx] + 1  # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background
    # sfd anchor matching strategy
    average_onestage = 6
    sort_overlaps, sort_id = overlaps.sort(1, descending=True)
    for gt_id in range(overlaps.size(0)):
        condition = best_truth_idx.eq(gt_id) * conf.byte()
        anchors_of_gt = condition.sum()
        if anchors_of_gt < average_onestage:  # set N as average number of anchor of each truths
            num_plus = 0
            for ac_id in range(priors.shape[0]):
                if sort_overlaps[gt_id][ac_id] < 0.1:
                    break
                elif not conf[sort_id[gt_id][ac_id]]:
                    # print (sort_overlaps[gt_id][ac_id] )
                    best_truth_idx[sort_id[gt_id][ac_id]] = gt_id
                    conf[sort_id[gt_id][ac_id]] = 1  # face is 1
                    num_plus += 1
                if num_plus == average_onestage - anchors_of_gt:
                    break
    matches = truths[best_truth_idx]  # Shape: [num_priors,4]
    loc = encode(matches, priors, variances)
    '''
    if part =='head':
        part_k = 1
    elif part =='body':
        part_k = 2
    if part in ['head','body']:
        #matches[:,0] = matches[:,0]-(1-2**part_k)/2*matches[:,2]
        #matches[:,1] = matches[:,1]-(1-2**part_k)/2*matches[:,3]
        #matches[:,2] = (2**part_k)*matches[:,2]
        #matches[:,3] = (2**part_k)*matches[:,3]
        loc[:,0] = loc[:,0]-(1-2**part_k)/2*loc[:,2]
        loc[:,1] = loc[:,1]-(1-2**part_k)/2*loc[:,3]
        loc[:,2] = (2**part_k)*loc[:,2]
        loc[:,3] = (2**part_k)*loc[:,3]
    '''
    # loc = encode(matches, priors, variances)
    loc_t[idx] = loc  # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


# average_anchor = []
def sfd_match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    '''
    S3FD: Single Shot Scale-invariant Face Detector
    '''
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    overlaps = jaccard(truths, point_form(priors))
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)

    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    conf = labels[best_truth_idx] + 1  # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background
    # sfd anchor matching strategy
    # swordli
    average_onestage = 6
    sort_overlaps, sort_id = overlaps.sort(1, descending=True)
    for gt_id in range(overlaps.size(0)):
        condition = best_truth_idx.eq(gt_id) * conf.byte()
        anchors_of_gt = condition.sum()
        if anchors_of_gt < average_onestage:  # set N as average number of anchor of each truths
            num_plus = 0
            for ac_id in range(priors.shape[0]):
                if sort_overlaps[gt_id][ac_id] < 0.1:
                    break
                elif not conf[sort_id[gt_id][ac_id]]:
                    # print (sort_overlaps[gt_id][ac_id] )
                    best_truth_idx[sort_id[gt_id][ac_id]] = gt_id
                    conf[sort_id[gt_id][ac_id]] = 1  # face is 1
                    num_plus += 1
                if num_plus == average_onestage - anchors_of_gt:
                    break
    matches = truths[best_truth_idx]  # Shape: [num_priors,4]
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc  # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[..., :2] + matched[..., 2:]) / 2 - priors[..., :2]
    # encode variance
    g_cxcy /= variances[0] * priors[..., 2:]
    # match wh / prior wh
    g_wh = (matched[..., 2:] - matched[..., :2]) / priors[..., 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], dim=-1)  # [num_priors,4]


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat(
        (
            priors[..., :2] + loc[..., :2] * variances[0] * priors[..., 2:],
            priors[..., 2:] * torch.exp(loc[..., 2:] * variances[1]),
        ),
        dim=-1,
    )
    boxes[..., :2] -= boxes[..., 2:] / 2
    boxes[..., 2:] += boxes[..., :2]
    # (cx,cy,w,h)->(x0,y0,x1,y1)
    return boxes


def bbox_overlaps_giou(bboxes1, bboxes2):
    """Calculate the gious between each bbox of bboxes1 and bboxes2.
    Args:
        bboxes1(Tensor): shape (n, 4)
        bboxes2(Tensor): shape (n, 4)
    Returns:
        gious(Tensor): shape (n,)
    """

    # bboxes1 = torch.FloatTensor(bboxes1)
    # bboxes2 = torch.FloatTensor(bboxes2)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return ious

    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])

    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])

    out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])

    out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_area = outer[:, 0] * outer[:, 1]
    union = area1 + area2 - inter_area
    closure = outer_area

    ious = inter_area / union - (closure - union) / closure
    ious = torch.clamp(ious, min=-1.0, max=1.0)

    return ious


def bbox_overlaps_diou(bboxes1, bboxes2):
    """Calculate the gious between each bbox of bboxes1 and bboxes2.
    Args:
        bboxes1(Tensor): shape (n, 4)
        bboxes2(Tensor): shape (n, 4)
    Returns:
        dious(Tensor): shape (n,)
    """
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = torch.zeros((rows, cols)).to(bboxes1.device)
    if rows * cols == 0:
        return ious

    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])

    out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]

    ctr_bbox1 = (bboxes1[:, :2] + bboxes1[:, 2:]) / 2.0
    ctr_bbox2 = (bboxes2[:, :2] + bboxes2[:, 2:]) / 2.0

    r_dious = (ctr_bbox1 - ctr_bbox2).pow(2).sum(dim=1) / (out_max_xy - out_min_xy).pow(2).sum(dim=1)

    union = area1 + area2 - inter_area

    ious = inter_area / union

    dious = ious - r_dious

    dious = dious.clamp(min=-1.0, max=1.0)

    # print(dious.shape)

    return dious


def bbox_overlaps_ciou(bboxes1, bboxes2):
    """Calculate the gious between each bbox of bboxes1 and bboxes2.
    Args:
        bboxes1(Tensor): shape (n, 4)
        bboxes2(Tensor): shape (n, 4)
    Returns:
        dious(Tensor): shape (n,)
    """

    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return ious

    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])

    out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]

    # outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    # outer_area = outer[:, 0] * outer[:, 1]

    ctr_bbox1 = (bboxes1[:, :2] + bboxes1[:, 2:]) / 2.0
    ctr_bbox2 = (bboxes2[:, :2] + bboxes2[:, 2:]) / 2.0

    r_dious = (ctr_bbox1 - ctr_bbox2).pow(2).sum(dim=1) / (out_max_xy - out_min_xy).pow(2).sum(dim=1)

    union = area1 + area2 - inter_area

    ious = inter_area / union

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    with torch.no_grad():
        arctan = torch.atan(w2 / h2) - torch.atan(w1 / h1)
        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
        S = 1 - ious
        alpha = v / (S + v)
        w_temp = 2 * w1
    ar = (8 / (math.pi ** 2)) * arctan * ((w1 - w_temp) * h1)

    cious = ious - r_dious - alpha * ar

    cious = cious.clamp(min=-1.0, max=1.0)

    return cious


class _ciou_op(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w_gt, h_gt, w, h):
        ctx.w_gt = w_gt
        ctx.h_gt = h_gt
        ctx.w = w
        ctx.h = h

        v = 4 / (np.pi ** 2) * (torch.atan2(w_gt, h_gt) - torch.atan2(w, h)) ** 2

        return v

    def backward(ctx, grad_output):
        w_gt, h_gt, w, h = ctx.w_gt, ctx.h_gt, ctx.w, ctx.h

        arc = 8 * (torch.atan2(w_gt, h_gt) - torch.atan2(w, h)) / (np.pi ** 2)

        return -h_gt * arc, w_gt * arc, h * arc, w * arc


def get_centerness_targets(loc_t, priors, variances, IoU=False):
    """generate the targets of centerness branch, like FCOS"""
    if not IoU:
        loc_t = decode(loc_t, priors, variances=variances)

    #
    l = priors[:, 0] - loc_t[:, 0]
    t = priors[:, 1] - loc_t[:, 1]
    r = loc_t[:, 2] - priors[:, 0]
    b = loc_t[:, 3] - priors[:, 1]

    left_right = torch.stack([l, r], dim=1)
    top_bottom = torch.stack([t, b], dim=1)

    centerness = torch.sqrt(
        left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0] * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    )
    
    assert not torch.isnan(centerness).any()
    return centerness


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


# nms for using ops in torchvision
def nms(boxes, scores, overlap=0.5, top_k=200):
    return python_nms(boxes, scores, overlap=overlap, top_k=top_k)


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def python_nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep, 0
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count
