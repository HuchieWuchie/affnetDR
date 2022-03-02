import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from lib.mask_rcnn import MaskRCNNPredictor, MaskAffordancePredictor, MaskRCNNHeads
import lib.mask_rcnn as mask_rcnn
import utils
import argparse
from PIL import Image
import numpy as np
import cv2
import time
from config import IITAFF
import os

def parse_args():

    parser = argparse.ArgumentParser(description='Perform inference on a given image using trained AffordanceNet')
    parser.add_argument('--gpu', dest='gpu',
                        help='GPU device id to use if not declared will use CPU',
                        action='store_true')
    parser.add_argument('--input', dest='input_path',
                        help='Path to rgb image',
                        default=None, type=str, required=True)
    parser.add_argument('--output', dest='output_path',
                        help='Prediction output folder',
                        default=None, type=str, required=True)
    parser.add_argument('--weights', dest='weights_path',
                        help='Pre-trained weights file',
                        default=None, type=str, required=True)
    parser.add_argument('--dataset', dest='dataset_name',
                        help='Dataset which the network was trained on [IIT-AFF, UMD, AFF-Synth]',
                        default=None, type=str, required=False)
    parser.add_argument('--object_threshold', dest='object_confidence_thresh',
                        help='Confidence threshold score for object detection, between 0 and 1, default = 0.9',
                        default=0.9, type=float)
    parser.add_argument('--affordance_threshold', dest='affordance_confidence_thresh',
                        help='Confidence threshold score for affordance segmentation, between 0 and 1, default = 0.4',
                        default=0.4, type=float)
    
    return parser.parse_args()

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


if __name__ == '__main__':

    args = parse_args()

    print()
    print("*************************************************************")
    print("********************* Inferring with ************************")
    print("*************************************************************")
    print("Input file: ", args.input_path)
    print("Output file: ", args.output_path)
    print("Weights file: ", args.weights_path)
    print("Dataset: ", args.dataset_name)
    print("Object detection confidence score threshold: ", args.object_confidence_thresh)
    print("Affordance segmentation confidence score threshold: ", args.affordance_confidence_thresh)
    if args.gpu:
        print("With GPU")
    else:
        print("With CPU")
    print("*************************************************************")

    aff_config = utils.get_dataset_config(args.dataset_name)    

    device = torch.device('cpu')
    if args.gpu:
        device = torch.device('cuda')
    
    num_classes = aff_config.NUM_CLASSES
    num_affordances = aff_config.NUM_AFFORDANCES
    
    model = utils.get_model_instance_segmentation(num_classes, num_affordances)
    model.load_state_dict(torch.load(args.weights_path))
    model.to(device)
    model.eval()

    if not os.path.isdir(args.output_path):
        os.mkdir(args.output_path)

    imgs = []
    if os.path.isdir(args.input_path):
        for f in os.listdir(args.input_path):
            imgs.append(os.path.join(args.input_path, f))
    else:
        imgs.append(args.input_path)

    for img_count, img_f in enumerate(imgs):
        img = Image.open(img_f).convert("RGB")
        width, height = img.size
        ratio = width / height

        img = img.resize((int(450 * ratio),450))
        img_vis = np.asarray(img).copy()
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
        x = [torchvision.transforms.ToTensor()(img).to(device)]

        ts = time.time() * 1000
        predictions = model(x)[0]
        te = time.time() * 1000
        print ("Prediction took: ", te - ts, " ms")
        boxes, labels, scores, masks = predictions['boxes'], predictions['labels'], predictions['scores'], predictions['masks']

        try:
            idx = scores > args.object_confidence_thresh
            labels = labels.cpu().detach().numpy()

            print(img_f.split(os.sep)[-1] + ": Found: \n")
            for label, score in zip(labels, scores):
                if score > 0.2:
                    print(aff_config.OBJ_CLASSES[label], score.item())

            boxes = boxes[idx].cpu().detach().numpy()
            labels = labels[idx.cpu().detach().numpy()]
            scores = scores[idx].cpu().detach().numpy()
            masks = masks[idx].cpu().detach().numpy()

            #idx.reverse()
            boxes = np.flip(boxes, axis = 0)
            labels = np.flip(labels, axis = 0)
            scores = np.flip(scores, axis = 0)
            masks= np.flip(masks, axis = 0)

            print()
            
            mask_full_vis = np.zeros(img_vis.shape).astype(np.uint8)
            for box, label, mask, score in zip(boxes, labels, masks, scores):
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                ps = (box[0], box[1])
                pe = (box[2], box[3])
                color = (0, 0, 255)
                thickness = 2
                img_vis = cv2.rectangle(img_vis, ps, pe, color, thickness)

                mask_pred = np.zeros(mask.shape)
                for count, m in enumerate(mask):
                    idx = m > args.affordance_confidence_thresh
                    mask_pred[count, idx] = m[idx]
                mask_arg = np.argmax(mask_pred, axis = 0)
                color_idxs = np.unique(mask_arg)
                #print(np.unique())
                
                for color_idx in color_idxs:
                    if color_idx != 0:
                        #img_vis[mask_arg == color_idx] = colors[color_idx]
                        mask_full_vis[mask_arg == color_idx] = aff_config.AFF_COLORS[color_idx]
                
                #print(np.unique(mask_pred))
                #mask_full_vis[mask_pred[2, : :] > 0.52] = aff_config.AFF_COLORS[2] # cut
                #mask_full_vis[mask_pred[3, : :] > 0.05] = aff_config.AFF_COLORS[3] # cut
                #mask_full_vis[mask_pred[5, : :] > 0.05] = aff_config.AFF_COLORS[5] # cut
                #mask_full_vis[mask_pred[6, : :] > 0.05] = aff_config.AFF_COLORS[6] # cut

                text = str(aff_config.OBJ_CLASSES[label]) + " " + str(round(score, 2))
                width = int(box[2] - box[0])
                fontscale = utils.get_optimal_font_scale(text, width)
                y_rb = int(box[3] + 40)
                img_vis[int(box[3]):y_rb, int(box[0])-1:int(box[2])+2] = (0, 0, 255)
                img_vis = cv2.putText(img_vis, text, (box[0], int(y_rb-10)), cv2.FONT_HERSHEY_SIMPLEX, fontscale, (255, 255, 255), 2, 1)
                
            img_vis = cv2.addWeighted(img_vis, 1.0, mask_full_vis, 0.5, 0)
            h, w = max(img_vis.shape[0], 400), img_vis.shape[1] + mask_full_vis.shape[1] + 400
            label_box = np.zeros((h, 400, 3))
            for count, (aff, color) in enumerate(zip(aff_config.AFF_CLASSES, aff_config.AFF_COLORS)):
                if count > 0:
                    x1 = 10
                    y1 = 30 * count + 10
                    x2 = 30
                    y2 = y1 + 20
                    label_box[y1:y2, x1:x2] = color

                    tx = x2 + 20
                    ty = y2
                    label_box = cv2.putText(label_box, aff, (tx, ty), fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                                            fontScale = 1, color = (255, 255, 255), thickness = 1, lineType = 1)

            full_vis = np.zeros((h, w, 3))
            full_vis[:img_vis.shape[0], :img_vis.shape[1]] = img_vis
            full_vis[:mask_full_vis.shape[0], img_vis.shape[1]:img_vis.shape[1] + mask_full_vis.shape[1]] = mask_full_vis
            full_vis[:, img_vis.shape[1] + mask_full_vis.shape[1]:w] = label_box
            
            f_name = os.path.join(args.output_path, img_f.split(os.sep)[-1].split('.')[0] + "_" + args.weights_path.split('.')[0] + "." + img_f.split(os.sep)[-1].split('.')[1])
            print(img_count+1, " / ", len(imgs), ": Saved " + f_name)
            cv2.imwrite(f_name, full_vis)
            
        except:
            pass
