import math
import sys
import time
import numpy as np
np.set_printoptions(suppress=True)
import argparse

import torch
import torchvision.models.detection.mask_rcnn

import lib.utils as mask_rcnn_utils
import utils
from dataset import InstanceSegmentationDataSet
import lib.mask_rcnn as mask_rcnn
import lib.transforms as T
import time
import os
import cv2

def parse_args():

    parser = argparse.ArgumentParser(description='Evaluate the affordance segmentation performance with a weighted F-measure')
    parser.add_argument('--gpu', dest='gpu',
                        help='GPU device id to use if not declared will use CPU',
                        action='store_true')
    parser.add_argument('--dataset_path', dest='dataset_path',
                        help='Path to root dataset directory',
                        default=None, type=str, required=True)
    parser.add_argument('--weights_path', dest='weights_path',
                        help='Path to .pth pretrained weights file',
                        default=None, type=str, required=True)
    parser.add_argument('--dataset_target', dest='dataset_target',
                        help='Dataset name to evaluate on [IIT-AFF, UMD, AFF-Synth]',
                        default=None, type=str, required=False)
    parser.add_argument('--dataset_source', dest='dataset_source',
                        help='Dataset name which the network was trained on [IIT-AFF, UMD, AFF-Synth]',
                        default=None, type=str, required=False)
    
    return parser.parse_args()

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    transforms.append(T.RandomHorizontalFlip(0.5))
    
    return T.Compose(transforms)

if __name__ == '__main__':

    args = parse_args()

    print()
    print("*************************************************************")
    print("********************* Evaluating with ***********************")
    print("*************************************************************")
    print("Dataset path: ", args.dataset_path)
    print("Weights file: ", args.weights_path)
    print("Dataset target: ", args.dataset_target)
    print("Dataset source: ", args.dataset_source)
    if args.gpu:
        print("With GPU")
    else:
        print("With CPU")
    print("*************************************************************")

    aff_config_source = utils.get_dataset_config(args.dataset_source)
    aff_config_target = utils.get_dataset_config(args.dataset_target)

    affordanceMap = utils.mapAffordancesTo(aff_config_source.AFF_CLASSES, aff_config_target.AFF_CLASSES)

    device = torch.device('cpu')
    if args.gpu:
        device = torch.device('cuda')
    
    num_classes = aff_config_source.NUM_CLASSES
    num_affordances = aff_config_source.NUM_AFFORDANCES

    model = utils.get_model_instance_segmentation(num_classes, num_affordances)
    model.load_state_dict(torch.load(args.weights_path))
    model.to(device)
    model.eval()

    dataset_test = aff_config_target.datasetLoader(root_dir = args.dataset_path, set = "test", transforms = get_transform(train=False), num_classes = num_classes, num_affordances = num_affordances)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=True, num_workers=1,
        collate_fn=mask_rcnn_utils.collate_fn)

    total_length = len(data_loader_test)
    fwb_scores = np.zeros(num_affordances)
    fwb_count = np.zeros(num_affordances)
    np.set_printoptions(precision=3)

    for count, x in enumerate(data_loader_test):
        
        gt_masks = x[1][0]['masks'].cpu().detach().numpy()
        img = x[0][0].cpu().detach().numpy()
        #print(img.shape)
        img = np.moveaxis(img, 0, 2)
        #img = np.reshape(img, (img.shape[1], img.shape[2], 3))
        img = img * 255
        x = [x[0][0].to(device)]
        predictions = model(x)[0]

        obj_thresh = 0.9
        aff_thresh = 0.1
        scores = predictions['scores'].cpu().detach().numpy()
        ix = scores > obj_thresh
        
        if True in ix:
          scores = scores[ix]
          masks = predictions['masks'].cpu().detach().numpy()[ix]
          boxes = predictions['boxes'].cpu().detach().numpy()[ix]
        else:
          scores = np.reshape(np.array(scores[0]), (1, 1))
          masks = predictions['masks'].cpu().detach().numpy()[0]
          masks = np.reshape(masks, (1, masks.shape[0], masks.shape[1], masks.shape[2]))
          boxes = predictions['boxes'].cpu().detach().numpy()[0]
          boxes = np.reshape(boxes, (1, 4))

        # outcomment pred_mask
        pred_mask = np.reshape(masks.max(0, keepdims=True), (aff_config_source.NUM_AFFORDANCES, masks.shape[-2], masks.shape[-1]))
        mask_full = np.zeros((aff_config_source.NUM_AFFORDANCES, img.shape[0], img.shape[1])).astype(np.uint8)
        for box, mask, score in zip(boxes, masks, scores):
          x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
          ps = (x1, y1)
          pe = (x2, y2)
          color = (0, 0, 255)
          thickness = 2
          img = cv2.rectangle(img, ps, pe, color, thickness)

          mask_pred = np.zeros(mask.shape)
          for countm, m in enumerate(mask):
              idx = m > aff_thresh
              mask_pred[countm, idx] = m[idx]
          mask_arg = np.argmax(mask_pred, axis = 0)
          color_idxs = np.unique(mask_arg)
        
        for aff_id in range(aff_config_source.NUM_AFFORDANCES):
          if aff_id != 0:
            if np.max(gt_masks[aff_id]):
              m_vis = np.zeros((mask_pred.shape[1], mask_pred.shape[2]))
              m_vis[mask_arg == aff_id] = 1
              fwb = utils.weighted_f_beta_score(m_vis, gt_masks[affordanceMap[aff_id]])
              fwb_scores[aff_id] += fwb
              fwb_count[aff_id] += 1

        fwb_mean = np.divide(fwb_scores, fwb_count)
        print(count + 1, " / ", total_length, " : ", np.mean(fwb_mean[~np.isnan(fwb_mean)]), fwb_mean)

                

                
