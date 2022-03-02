import numpy as np
import time
import os
import cv2
import scipy
import scipy.ndimage
from lib.mask_rcnn import MaskRCNNPredictor, MaskAffordancePredictor, MaskRCNNHeads
from lib.faster_rcnn import FastRCNNPredictor
import lib.mask_rcnn as mask_rcnn
import config

def weighted_f_beta_score(candidate, gt, beta=1.0):
    """
    from https://gist.github.com/egparedes/b3aa721381a889e3be590711f979688d
    Compute the Weighted F-beta measure (as proposed in "How to Evaluate Foreground Maps?" [Margolin et al. - CVPR'14])
    Original MATLAB source code from:
        [https://cgm.technion.ac.il/Computer-Graphics-Multimedia/Software/FGEval/resources/WFb.m]
    :param candidate: FG - Binary/Non binary candidate map with values in the range [0-1]
    :param gt: Binary ground truth map
    :param beta: attach 'beta' times as much importance to Recall as to Precision (default=1)
    :result: the Weighted F-beta score
    """

    if np.min(candidate) < 0.0 or np.max(candidate) > 1.0:
        raise ValueError("'candidate' values must be inside range [0 - 1]")

    if gt.dtype in [np.bool, np.bool_, np.bool8]:
        gt_mask = gt
        not_gt_mask = np.logical_not(gt_mask)
        gt = np.array(gt, dtype=candidate.dtype)
    else:
        if not np.all(np.isclose(gt, 0) | np.isclose(gt, 1)):
            raise ValueError("'gt' must be a 0/1 or boolean array")
        gt_mask = np.isclose(gt, 1)
        not_gt_mask = np.logical_not(gt_mask)
        gt = np.asarray(gt, dtype=candidate.dtype)

    E = np.abs(candidate - gt)
    dist, idx = scipy.ndimage.morphology.distance_transform_edt(not_gt_mask, return_indices=True)

    # Pixel dependency
    Et = np.array(E)
    # To deal correctly with the edges of the foreground region:
    Et[not_gt_mask] = E[idx[0, not_gt_mask], idx[1, not_gt_mask]]
    sigma = 5.0
    EA = scipy.ndimage.gaussian_filter(Et, sigma=sigma, truncate=3 / sigma,
                                       mode='constant', cval=0.0)
    min_E_EA = np.minimum(E, EA, where=gt_mask, out=np.array(E))

    # Pixel importance
    B = np.ones(gt.shape)
    B[not_gt_mask] = 2 - np.exp(np.log(1 - 0.5) / 5 * dist[not_gt_mask])
    Ew = min_E_EA * B

    # Final metric computation
    eps = np.spacing(1)
    TPw = np.sum(gt) - np.sum(Ew[gt_mask])
    FPw = np.sum(Ew[not_gt_mask])
    R = 1 - np.mean(Ew[gt_mask])  # Weighed Recall
    P = TPw / (eps + TPw + FPw)  # Weighted Precision

    # Q = 2 * (R * P) / (eps + R + P)  # Beta=1
    Q = (1 + beta**2) * (R * P) / (eps + R + (beta * P))

    return Q

def get_model_instance_segmentation(num_classes, num_affordances):
    # load an instance segmentation model pre-trained on COCO
    model = mask_rcnn.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace mask branch
    mask_layers = [256]
    mask_dilation = 1
    out_channels = model.backbone.out_channels
    model.roi_heads.mask_head = MaskRCNNHeads(out_channels, mask_layers, mask_dilation)
    
    # and replace the mask predictor with a new one
    in_features_mask = 256
    hidden_layer = 128
    #model.roi_heads.mask_predictor = MaskRCNNPredictor(256,
    #                                                   256,
    #                                                   num_affordances)
    
    model.roi_heads.mask_predictor = MaskAffordancePredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_affordances)

    return model

def get_latest_epoch(chkp_folder):
    """Gets the latest epoch file path
        input:  chkp_folder : str, folder containing .pth files,
                                    assumes naming 0.pth, 1.pth ...
        output: epoch_no    : int, latest epoch completed
                chkp_path   : str, relative path to latest checkpoint
    """

    files = os.listdir(chkp_folder)
    if len(files):
        files = [x.replace('.pth', '') for x in files]
        files = sorted(files, key = int)
        
        epoch_no = int(files[-1])
        chkp_path = os.path.join(chkp_folder, files[-1] + ".pth")
        return epoch_no, chkp_path
    
    return 0, None

def get_optimal_font_scale(text, width):
    """https://stackoverflow.com/questions/52846474/how-to-resize-text-for-cv2-puttext-according-to-the-image-size-in-opencv-python"""
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
        new_width = textSize[0][0]
        if (new_width <= width):
            if scale/10 > 1.0:
                return 1
            return scale/10
    return 1

def get_dataset_config(dataset_name):
    dataset_classes = config.get_classes()
    for c in dataset_classes:
        if dataset_name == c.NAME and c.NAME is not None:
            return c()
    
    print(dataset_name, " is not a valid dataset name, valid ones are:\n")
    for name in config.get_class_names():
        print(name)
    return None

def mapAffordancesTo(source, target):

    target_idxs = []
    for affordance in source:
        target_idx = -1
        if affordance in target:
            target_idx = target.index(affordance)
        target_idxs.append(target_idx)
    return target_idxs


if __name__ == '__main__':
    config.get_class_names()