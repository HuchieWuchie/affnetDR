import math
import sys
import time
import numpy as np
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

def parse_args():

    parser = argparse.ArgumentParser(description='Train a Mask-RCNN for affordance segmentation')
    parser.add_argument('--gpu', dest='gpu',
                        help='GPU device id to use if not declared will use CPU',
                        action='store_true')
    parser.add_argument('--iterations', dest='num_iterations',
                        help='Number of iterations to train, default=300000',
                        default=300000, type=int)
    parser.add_argument('--batch_size', dest='batch_size',
                        help='Number of images in a batch to train, default=1',
                        default=1, type=int)
    parser.add_argument('--dataset_path', dest='dataset_path',
                        help='Path to root dataset directory',
                        default=None, type=str, required=True)
    parser.add_argument('--output_path', dest='output_path',
                        help='Path to output logs, weights and checkpoints',
                        default=None, type=str, required=True)
    parser.add_argument('--dataset', dest='dataset_name',
                        help='Dataset name [IIT-AFF, UMD, Synth-Aff]',
                        default=None, type=str, required=False)
    parser.add_argument('--skip_evaluation', dest='skip_eval',
                        help='Skip evaluation after each epoch, and only evaluate at the end.',
                        action='store_true')
    parser.add_argument('--restart', dest='restart_train',
                        help='Will continue from checkpoint in specified output_path if not specified.',
                        action='store_true')
    
    return parser.parse_args()


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, current_iteration, iteration_checkpoint_size, max_iterations, chkp_folder,scaler=None):
    model.train()
    metric_logger = mask_rcnn_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", mask_rcnn_utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        current_iteration += 1;
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = mask_rcnn_utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()
        

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if current_iteration % iteration_checkpoint_size == 0:
          checkpoint_path = os.path.join(chkp_folder, str(current_iteration) + ".pth")
          torch.save(model.state_dict(), checkpoint_path)
        if current_iteration > max_iterations:
          return metric_logger, current_iteration, False

    return metric_logger, current_iteration, True


@torch.inference_mode()
def validation_loss(model, data_loader, device, print_freq=100):
    #n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    model.train()
    metric_logger = mask_rcnn_utils.MetricLogger(delimiter="  ")
    header = "Test:"

    losses = []

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        model_time = time.time()
        loss_dict = model(images, targets)

        model_time = time.time() - model_time

        evaluator_time = time.time()
        loss_dict_reduced = mask_rcnn_utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        losses.append(loss_value)

        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    mean_loss = np.mean(np.array(losses))
    print("Averaged stats:", mean_loss)

    return mean_loss

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    transforms.append(T.RandomHorizontalFlip(0.5))
    
    return T.Compose(transforms)

if __name__ == '__main__':

    args = parse_args()

    print()
    print("*************************************************************")
    print("********************* TRAINING WITH *************************")
    print("*************************************************************")
    print("Dataset path: ", args.dataset_path)
    print("Output path: ", args.output_path)
    print("Dataset: ", args.dataset_name)
    if args.gpu:
        print("With GPU")
    else:
        print("With CPU")
    print("Number of iterations: ", args.num_iterations)
    print("Batch size: ", args.batch_size)
    print("Skip evaluation step: ", args.skip_eval)
    print("Restart training: ", args.restart_train)
    print("*************************************************************")

    aff_config = utils.get_dataset_config(args.dataset_name)

    device = torch.device('cpu')
    if args.gpu:
        device = torch.device('cuda')
    
    num_classes = aff_config.NUM_CLASSES
    num_affordances = aff_config.NUM_AFFORDANCES

    model = utils.get_model_instance_segmentation(num_classes, num_affordances)
    model.to(device)

    log_folder = args.output_path
    chkp_folder = os.path.join(args.output_path, "checkpoints")
    weights_path = os.path.join(args.output_path, "weights.pth")
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    if not os.path.exists(chkp_folder):
        os.mkdir(chkp_folder)

    current_epoch = 0
    if not args.restart_train:
        last_epoch, latest_chkp_path = utils.get_latest_epoch(chkp_folder)
        current_epoch = last_epoch + 1
        if latest_chkp_path is not None:
            model.load_state_dict(torch.load(latest_chkp_path))
            print("Continuing from latest epoch: ", last_epoch, latest_chkp_path)
        
        else:
            print("Found no previous checkpoints, starting from scratch...")

    for param in model.backbone.parameters():
        param.requires_grad = False
        
    # use our dataset and defined transformations
    dataset = aff_config.datasetLoader(root_dir = args.dataset_path, set = "train", transforms = get_transform(train=True), num_classes = num_classes, num_affordances = num_affordances)
    dataset_test = aff_config.datasetLoader(root_dir = args.dataset_path, set = "test", transforms = get_transform(train=False), num_classes = num_classes, num_affordances = num_affordances)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.001)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=1,
        collate_fn=mask_rcnn_utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=mask_rcnn_utils.collate_fn)

    current_iteration = 0
    continue_training = True
    min_val_loss = None

    for epoch in range(current_epoch, args.num_iterations):
        _, current_iteration, continue_training = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1000, current_iteration = current_iteration, iteration_checkpoint_size=10000/args.batch_size, max_iterations = args.num_iterations, chkp_folder = chkp_folder)
        checkpoint_path = os.path.join(chkp_folder, str(epoch) + ".pth")
        torch.save(model.state_dict(), weights_path)

        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        if not args.skip_eval:
            val_loss = validation_loss(model, data_loader_test, device=device, print_freq = 100)
            
            print("Mean validation loss: ", val_loss)

            if min_val_loss is None:
                min_val_loss = val_loss

            if val_loss < min_val_loss:
                print("Model improved, saving parameters...")
                min_val_loss = val_loss
                torch.save(model.state_dict(), weights_path)
        if continue_training == False:
          break
