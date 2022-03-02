import torch
import os
import numpy as np
from PIL import Image
import cv2
import torchvision.transforms
import scipy.io

class InstanceSegmentationDataSet(torch.utils.data.Dataset):
    def __init__(self, root_dir, transforms, num_classes, num_affordances, set=""):

        assert set in ["train", "test"]

        self.root_dir = os.path.join(root_dir, set)
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root_dir, "rgb"))))
        self.masks = list(sorted(os.listdir(os.path.join(root_dir, "masks"))))
        self.objects = list(sorted(os.listdir(os.path.join(root_dir, "object_labels"))))
        self.num_classes = num_classes
        self.num_affordances = num_affordances

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,
                    idx: int):
        
        img_path = os.path.join(self.root_dir, "rgb", self.imgs[idx])
        mask_path = os.path.join(self.root_dir, "masks", self.masks[idx])
        object_path = os.path.join(self.root_dir, "object_labels", self.objects[idx])

        img = Image.open(img_path).convert("RGB")

        mask_full_one_channel = np.loadtxt(mask_path).astype(np.uint8)

        h, w = mask_full_one_channel.shape
        h_scale, w_scale = 1, 1

        if self.rescale:
            
            if h < 1024 or w < 1024:
                h_scale = 1024 / h
                w_scale = 1024 / w
                img = img.resize((1024,1024))
                mask_full_one_channel = cv2.resize(mask_full_one_channel, (1024,1024))

        objects = np.loadtxt(object_path).astype(np.int32)
        objects = np.reshape(objects, (-1, 5))
        no_instances = objects.shape[0]
        #mask_full = np.zeros((no_instances, mask_full_one_channel.shape[0], mask_full_one_channel.shape[1])).astype(np.uint8)
        mask_full = np.zeros((self.num_affordances, mask_full_one_channel.shape[0], mask_full_one_channel.shape[1])).astype(np.uint8)
        mask_full[0, :, :] = 1 # set all background pixels to 1
        class_ids = []
        bboxes = []

        for i, obj in enumerate(objects):
            class_id, x1, y1, x2, y2 = obj
            x1, y1, x2, y2 = int(x1 * w_scale), int(y1 * h_scale), int(x2 * w_scale), int(y2 * h_scale)
            class_id = class_id + 1
            for j in range(self.num_affordances):
                
                # dont update background
                if j > 0:
                    mask_full[j, y1:y2, x1:x2] = mask_full_one_channel[y1:y2, x1:x2] == j
                    mask_full[0, y1:y2, x1:x2] = 0 # set background pixels to negative

            class_ids.append(class_id)
            bboxes.append([x1, y1, x2, y2])
         
        # convert everything into a torch.Tensor
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(class_ids, dtype=torch.int64)
        masks = torch.as_tensor(mask_full, dtype=torch.uint8)
        
        image_id = torch.tensor([idx])
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:,0])

        iscrowd = torch.zeros((len(class_ids),), dtype=torch.int64)

        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # Preprocessing
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        #img = torchvision.transforms.Resize((1024,1024), img)
        return img, target


class InstanceSegmentationDataSetObjectness(torch.utils.data.Dataset):
    def __init__(self, root_dir, transforms, num_classes, num_affordances, set = ""):

        assert set in ["train", "test"]

        self.root_dir = os.path.join(root_dir, set)
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(self.root_dir, "rgb"))))
        self.masks = list(sorted(os.listdir(os.path.join(self.root_dir, "masks"))))
        self.objects = list(sorted(os.listdir(os.path.join(self.root_dir, "object_labels"))))
        self.num_classes = num_classes
        self.num_affordances = num_affordances

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,
                    idx: int):
        
        img_path = os.path.join(self.root_dir, "rgb", self.imgs[idx])
        mask_path = os.path.join(self.root_dir, "masks", self.masks[idx])
        object_path = os.path.join(self.root_dir, "object_labels", self.objects[idx])

        img = Image.open(img_path).convert("RGB")

        mask_full_one_channel = np.loadtxt(mask_path).astype(np.uint8)

        h, w = mask_full_one_channel.shape
        h_scale, w_scale = 1, 1

        if self.rescale:
            
            if h < 1024 or w < 1024:
                h_scale = 1024 / h
                w_scale = 1024 / w
                img = img.resize((1024,1024))
                mask_full_one_channel = cv2.resize(mask_full_one_channel, (1024,1024))

        objects = np.loadtxt(object_path).astype(np.int32)
        objects = np.reshape(objects, (-1, 5))
        no_instances = objects.shape[0]
        #mask_full = np.zeros((no_instances, mask_full_one_channel.shape[0], mask_full_one_channel.shape[1])).astype(np.uint8)
        mask_full = np.zeros((self.num_affordances, mask_full_one_channel.shape[0], mask_full_one_channel.shape[1])).astype(np.uint8)
        mask_full[0, :, :] = 1 # set all background pixels to 1
        class_ids = []
        bboxes = []

        for i, obj in enumerate(objects):
            _, x1, y1, x2, y2 = obj
            x1, y1, x2, y2 = int(x1 * w_scale), int(y1 * h_scale), int(x2 * w_scale), int(y2 * h_scale)
            class_id = 1
            #mask_full[class_id, y1:y2, x1:x2] = mask_full_one_channel[y1:y2, x1:x2] > 0
            #mask_full[i, y1:y2, x1:x2] = mask_full_one_channel[y1:y2, x1:x2] > 0
            for j in range(self.num_affordances):
                
                # dont update background
                if j > 0:
                    mask_full[j, y1:y2, x1:x2] = mask_full_one_channel[y1:y2, x1:x2] == j
                    mask_full[0, y1:y2, x1:x2] = 0 # set background pixels to negative

            class_ids.append(class_id)
            bboxes.append([x1, y1, x2, y2])
        #print(mask_full.shape)
        #mask_full = mask_full * 255
        """
        p = 0
        for m in mask_full:
            mc = m.copy()
            
            p += 1
            for b in bboxes:
                x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                mc = cv2.rectangle(mc, (x1, y1), (x2, y2), color = 255, thickness = 2)
            cv2.imwrite("test/" + str(p) + ".jpg", mc)
        """
            
        # convert everything into a torch.Tensor
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(class_ids, dtype=torch.int64)
        masks = torch.as_tensor(mask_full, dtype=torch.uint8)
        
        image_id = torch.tensor([idx])
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:,0])

        iscrowd = torch.zeros((len(class_ids),), dtype=torch.int64)

        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # Preprocessing
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        #img = torchvision.transforms.Resize((1024,1024), img)
        return img, target

class UMDCategoryDataSet(torch.utils.data.Dataset):
    def __init__(self, root_dir, transforms, num_classes, num_affordances, set=""):

        self.root_dir = root_dir
        self.transforms = transforms
        self.num_classes = num_classes
        self.num_affordances = num_affordances
        self.split_int = 0
        self.imgs = []
        self.masks = []
        self.labels = []

        OBJ_CLASSES = ('__background__', 'knife', 'saw', 'scissors', 'shears', 'scoop',
                                        'spoon', 'trowel', 'bowl', 'cup', 'ladle',
                                        'mug', 'pot', 'shovel', 'turner', 'hammer',
                                        'mallet', 'tenderizer')

        assert set in ["train", "test"]
        if set == "train":
            self.split_int = 1
        elif set == "test":
            self.split_int = 2
        
        meta_file = os.path.join(root_dir, "category_split.txt")
        with open(meta_file) as f:
            lines = f.readlines()
        
        img_folders = []
        for line in lines:
            line = line.replace('\n', '')
            category_int = int(line.split(' ')[0])
            folder_name = line.split(' ')[1]

            if category_int == self.split_int:
                img_folders.append(folder_name)
        
        for folder in img_folders:
            current_folder = os.path.join(self.root_dir, "tools", folder)
            files = os.listdir(current_folder)
            for file in files:
                extension = os.path.splitext(file)[1]
                file_name = os.path.splitext(file)[0]
                if extension == ".png" or extension == ".jpg":
                    file_split = file_name.split('_')
                    if "rgb" in file_split:
                        self.imgs.append(os.path.join(current_folder, file))
                        self.masks.append(os.path.join(current_folder, file_name.replace('_rgb', '_label.mat')))
                        object_name = file_split[0]
                        self.labels.append(OBJ_CLASSES.index(object_name))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,
                    idx: int):
        
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]

        img = Image.open(img_path).convert("RGB")

        mask_full_one_channel = scipy.io.loadmat(mask_path)['gt_label']

        mask_full = np.zeros((self.num_affordances, mask_full_one_channel.shape[0], mask_full_one_channel.shape[1])).astype(np.uint8)
        mask_full[0, :, :] = 1 # set all background pixels to 1
        label = [self.labels[idx]]
        bbox = []

        # compute bbox
        occupied_pixels = mask_full_one_channel > 0
        occupied_pixels_idxs = np.where(occupied_pixels)
        x1, x2 = np.min(occupied_pixels_idxs[1]), np.max(occupied_pixels_idxs[1])
        y1, y2 = np.min(occupied_pixels_idxs[0]), np.max(occupied_pixels_idxs[0])

        for j in range(self.num_affordances):

            # dont update background
            if j > 0:
                mask_full[j, y1:y2, x1:x2] = mask_full_one_channel[y1:y2, x1:x2] == j
                mask_full[0, y1:y2, x1:x2] = 0 # set background pixels to negative

        bbox.append([x1, y1, x2, y2])
            
        # convert everything into a torch.Tensor
        bbox = torch.as_tensor(bbox, dtype=torch.float32)
        label = torch.as_tensor(label, dtype=torch.int64)
        masks = torch.as_tensor(mask_full, dtype=torch.uint8)
        
        image_id = torch.tensor([idx])
        area = (bbox[:, 3] - bbox[:, 1]) * (bbox[:, 2] - bbox[:,0])

        iscrowd = torch.zeros((len(label),), dtype=torch.int64)

        target = {}
        target["boxes"] = bbox
        target["labels"] = label
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # Preprocessing
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        #img = torchvision.transforms.Resize((1024,1024), img)
        return img, target


class AFFSynthDataSet(torch.utils.data.Dataset):
    def __init__(self, root_dir, transforms, num_classes, num_affordances, set=""):

        assert set in ["train", "test"]

        self.root_dir = os.path.join(root_dir, set)
        self.transforms = transforms
        self.set = set
        self.imgs = list(sorted(os.listdir(os.path.join(self.root_dir, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(self.root_dir, "masks"))))
        self.objects = list(sorted(os.listdir(os.path.join(self.root_dir, "object_labels"))))
        print(len(self.imgs), len(self.masks), len(self.objects))
        
        self.num_classes = num_classes
        self.num_affordances = num_affordances

        C_BACKGROUND = (0,0,0)
        C_GRASP = (0, 0, 255)
        C_CUT = (0, 255, 0)
        C_SCOOP = (123,255, 123)
        C_CONTAIN = (255, 0, 0)
        C_POUND = (255, 255, 0)
        C_SUPPORT = (255, 255, 255)
        C_WRAPGRASP = (255, 0, 255)
        C_DISPLAY = (122, 122, 122)
        C_ENGINE = (0, 255, 255)
        C_HIT = (70, 70, 70)

        self.AFF_COLORS = [C_BACKGROUND, C_GRASP, C_CUT, C_SCOOP, C_CONTAIN,
                    C_POUND, C_SUPPORT, C_WRAPGRASP, C_DISPLAY, C_ENGINE, C_HIT]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,
                    idx: int):
        
        img_path = os.path.join(self.root_dir, "images", self.imgs[idx])
        mask_path = os.path.join(self.root_dir, "masks", self.masks[idx])
        object_path = os.path.join(self.root_dir, "object_labels", self.objects[idx])

        img = Image.open(img_path).convert("RGB")

        objects = []
        with open(object_path) as f:
            lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            class_id, x_min, y_min, x_max, y_max = line.split(' ')
            objects.append([int(class_id), int(x_min), int(y_min), int(x_max), int(y_max)])
        
        objects = np.array(objects).astype(int)
        objects = np.reshape(objects, (-1, 5))

        mask_rgb = Image.open(mask_path)
        mask_rgb.load()
        mask_rgb = np.array(mask_rgb)
        
        mask_full = np.zeros((self.num_affordances, mask_rgb.shape[0], mask_rgb.shape[1])).astype(np.uint8)
        class_ids = []
        bboxes = []

        for j in range(self.num_affordances):

            idx = mask_rgb[:, :][:] == self.AFF_COLORS[j]
            idx = np.sum(idx, axis = -1) == 3
            mask_full[j, idx] = 1

        for i, obj in enumerate(objects):
            class_id, x1, y1, x2, y2 = obj
            class_ids.append(class_id)
            bboxes.append([x1, y1, x2, y2])
         
        # convert everything into a torch.Tensor
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(class_ids, dtype=torch.int64)
        masks = torch.as_tensor(mask_full, dtype=torch.uint8)
        
        image_id = torch.tensor([idx])
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:,0])

        iscrowd = torch.zeros((len(class_ids),), dtype=torch.int64)

        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # Preprocessing
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        #img = torchvision.transforms.Resize((1024,1024), img)
        return img, target

class AFFSynthDataSetCollapsed(torch.utils.data.Dataset):
    def __init__(self, root_dir, transforms, num_classes, num_affordances, set=""):

        assert set in ["train", "test"]

        self.root_dir = os.path.join(root_dir, set)
        self.transforms = transforms
        self.set = set
        self.imgs = list(sorted(os.listdir(os.path.join(self.root_dir, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(self.root_dir, "masks"))))
        self.objects = list(sorted(os.listdir(os.path.join(self.root_dir, "object_labels"))))
        print(len(self.imgs), len(self.masks), len(self.objects))
        
        self.num_classes = num_classes
        self.num_affordances = num_affordances

        C_BACKGROUND = (0,0,0)
        C_GRASP = (0, 0, 255)
        C_CUT = (0, 255, 0)
        C_SCOOP = (123,255, 123)
        C_CONTAIN = (255, 0, 0)
        C_POUND = (255, 255, 0)
        C_SUPPORT = (255, 255, 255)
        C_WRAPGRASP = (255, 0, 255)
        C_DISPLAY = (122, 122, 122)
        C_ENGINE = (0, 255, 255)
        C_HIT = (70, 70, 70)

        self.AFF_COLORS = [C_BACKGROUND, C_GRASP, C_CUT, C_SCOOP, C_CONTAIN,
                    C_POUND, C_SUPPORT, C_WRAPGRASP, C_DISPLAY, C_ENGINE, C_HIT]
        self.CLASS_TO_DATSET_CLASS = {0: 0,
                          1: 2,
                          2: 2,
                          3: 2,
                          4: 2,
                          5: 3,
                          6: 3,
                          7: 3,
                          8: 4,
                          9: 4,
                          10: 4,
                          11: 4,
                          12: 4,
                          13: 6,
                          14: 6,
                          15: 5,
                          16: 5,
                          17: 5,
                          18: 1,
                          19: 9,
                          20: 8,
                          21: 4,
                          22: 10
                          }


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,
                    idx: int):
        
        img_path = os.path.join(self.root_dir, "images", self.imgs[idx])
        mask_path = os.path.join(self.root_dir, "masks", self.masks[idx])
        object_path = os.path.join(self.root_dir, "object_labels", self.objects[idx])

        img = Image.open(img_path).convert("RGB")

        objects = []
        with open(object_path) as f:
            lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            class_id, x_min, y_min, x_max, y_max = line.split(' ')
            objects.append([int(class_id), int(x_min), int(y_min), int(x_max), int(y_max)])
        
        objects = np.array(objects).astype(int)
        objects = np.reshape(objects, (-1, 5))

        mask_rgb = Image.open(mask_path)
        mask_rgb.load()
        mask_rgb = np.array(mask_rgb)
        
        mask_full = np.zeros((self.num_affordances, mask_rgb.shape[0], mask_rgb.shape[1])).astype(np.uint8)
        #mask_full[0, :, :] = 1 # set all background pixels to 1
        class_ids = []
        bboxes = []

        for j in range(self.num_affordances):

            idx = mask_rgb[:, :][:] == self.AFF_COLORS[j]
            idx = np.sum(idx, axis = -1) == 3
            mask_full[j, idx] = 1

        for i, obj in enumerate(objects):
            class_id, x1, y1, x2, y2 = obj
            area = (x2 - x1) * (y2 - y1)
            if area > 20:
              class_id = self.CLASS_TO_DATSET_CLASS[class_id]
              class_ids.append(class_id)
              bboxes.append([x1, y1, x2, y2])
            else:
              class_ids.append(0)
              bboxes.append([0, 0, 5, 5])
            
         
        # convert everything into a torch.Tensor
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(class_ids, dtype=torch.int64)
        masks = torch.as_tensor(mask_full, dtype=torch.uint8)
        
        image_id = torch.tensor([idx])
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:,0])

        iscrowd = torch.zeros((len(class_ids),), dtype=torch.int64)

        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # Preprocessing
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        #img = torchvision.transforms.Resize((1024,1024), img)
        return img, target
