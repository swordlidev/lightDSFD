# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = (
    (255, 0, 0, 128),
    (0, 255, 0, 128),
    (0, 0, 255, 128),
    (0, 255, 255, 128),
    (255, 0, 255, 128),
    (255, 255, 0, 128),
)

MEANS = (104, 117, 123)

widerface_640 = {
    "num_classes": 2,
    #'lr_steps': (80000, 100000, 120000),
    #'max_iter': 120000,
    "feature_maps": [160, 80, 40, 20, 10, 5],
    "min_dim": 640,
    # "feature_maps": [256, 128, 64, 32, 16, 8],
    # "min_dim": 1024,
    "steps": [4, 8, 16, 32, 64, 128],  # stride
    "variance": [0.1, 0.2],
    "clip": True,  # make default box in [0,1]
    "name": "WIDERFace",
    "l2norm_scale": [10, 8, 5],
    "base": [64, 64, "M", 128, 128, "M", 256, 256, 256, "C", 512, 512, 512, "M", 512, 512, 512,],
    "extras": [256, "S", 512, 128, "S", 256],
    "mbox": [1, 1, 1, 1, 1, 1],
    #'mbox': [2, 2, 2, 2, 2, 2],
    #'mbox': [4, 4, 4, 4, 4, 4],
    "min_sizes": [16, 32, 64, 128, 256, 512],
    "max_sizes": [],
    #'max_sizes': [8, 16, 32, 64, 128, 256],
    #'aspect_ratios': [ [],[],[],[],[],[] ],   # [1,2]  default 1
    "aspect_ratios": [[1.5], [1.5], [1.5], [1.5], [1.5], [1.5]],  # [1,2]  default 1
    "backbone": "resnet18",  # vgg, resnet, detnet, resnet50
    # about PC-DARTS
    'edge_normalization': True,
    'groups': 4,
    "lr_steps": (50, 80, 100, 121),
    "max_epochs": 121,
    
    'STC_STR': False,
    'auxiliary_classify': False,
    'retrain_with_bn': True,
    'residual_learning': True,
    'syncBN': False,
    
    'GN': False,
    
    # BiFPN
    "bidirectional_feature_pyramid_network": True,
    # FPN
    "feature_pyramid_network": False,
    # whether to search FPN
    "search_feature_pyramid_network": True,
    #
    "use_searched_feature_pyramid_network": False,
    # which layer to fed into the FPN cell
    "inter_input_nums": [2, 3, 3, 3, 3, 2],
    "out_skip_input_nums": [0, 0, 0, 0, 0, 0],
    "inter_start_layer": [0, 0, 1, 2, 3, 4],
    "out_skip_start_layer": [0, 1, 2, 3, 4, 5],
    "bottom_up_path": False,
    
    # CPM
    "cpm_simple": False,
    "cpm_simple_v2": True,
    "context_predict_module": False,
    "search_context_predict_module": False,
    
    "cross_stack": False,
    "fpn_cpm_channel": 128,     #256
    "stack_convs": 1,       #3
    
    "max_in_out": True,
    "improved_max_in_out": False,
    
    "FreeAnchor": False,
    "GHM": False,
    
    "margin_loss_type": "",     # arcface, cosface, arcface_scale, cosface_scale
    "margin_loss_s": 1,
    "margin_loss_m": 0.2,
    "focal_loss": False,
    "iou_loss": "",     # giou, diou, ciou
    "ATSS": False,
    "ATSS_topk": 9,
    "centerness": False,
    
    "pyramid_anchor": True,
    "refinedet": False,
    "max_out": False,
    "anchor_compensation": False,
    "data_anchor_sampling": False,
    "overlap_thresh": [0.4],
    "negpos_ratio": 3,
    # test
    "nms_thresh": 0.3,
    "conf_thresh": 0.05,  # 0.01
    "num_thresh": 2000,  # 5000
}
