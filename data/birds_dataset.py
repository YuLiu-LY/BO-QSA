import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

from typing import Tuple
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from methods.utils import rescale


class CubDataset(Dataset):
    def __init__(
        self, 
        data_root: str,
        resolution: Tuple[int, int],
        data_split='train', 
        use_rescale=False, # rescale to [-1, 1]
        use_flip=False,
    ):
        super().__init__()        
        self.data_split = data_split
        self.use_flip = use_flip

        self.transform_seg = transforms.Compose([
            transforms.Resize(resolution, interpolation=Image.NEAREST),            
            transforms.ToTensor(),
        ])
        trans = [
            transforms.Resize(resolution),            
            transforms.ToTensor(),
        ]
        if use_rescale:
            trans.append(transforms.Lambda(rescale))
        self.transform = transforms.Compose(trans)
        
        self.ROOT_DIR = data_root

        self.bbox_meta, self.file_meta = self.collect_meta()

    def __len__(self):
        return len(self.file_meta)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: sample # {}'.format(index))
            item = self.load_item(0)

        return {
            'image': item[0],
            'mask': item[1], 
        }   

    def collect_meta(self):
        """ Returns a dictionary with image filename as 
        'key' and its bounding box coordinates as 'value' """

        data_dir = self.ROOT_DIR

        bbox_path = os.path.join(data_dir, 'bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)

        filepath = os.path.join(data_dir, 'images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)

        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)

        splits = np.loadtxt(os.path.join(data_dir, 'train_val_test_split.txt'), int)

        for i in range(0, numImgs):            
            bbox = df_bounding_boxes.iloc[i][1:].tolist()
            key = filenames[i][:-4]
            filename_bbox[key] = bbox

        filenames = [fname[:-4] for fname in filenames]

        if self.data_split == 'train': # training split
            filenames = np.array(filenames)
            filenames = filenames[splits[:, 1] == 0]
            filename_bbox_ = {fname: filename_bbox[fname] for fname in filenames}
        elif self.data_split == 'val': # validation split
            filenames = np.array(filenames)
            filenames = filenames[splits[:, 1] == 1]
            filename_bbox_ = {fname: filename_bbox[fname] for fname in filenames}
        elif self.data_split == 'test': # testing split
            filenames = np.array(filenames)
            filenames = filenames[splits[:, 1] == 2]
            filename_bbox_ = {fname: filename_bbox[fname] for fname in filenames}

        print('Filtered filenames: ', len(filenames))
        return filename_bbox_, filenames

    def load_item(self, index):
        key = self.file_meta[index]        
        bbox = self.bbox_meta[key]
        
        data_dir = self.ROOT_DIR
        
        img_path = '%s/images/%s.jpg' % (data_dir, key)
        img = self.load_imgs(img_path, bbox)

        seg_path = '%s/segmentations/%s.png' % (data_dir, key)
        seg = self.load_segs(seg_path, bbox)

        if self.use_flip and np.random.uniform() > 0.5:
            img = torch.flip(img, dims=[-1])
            seg = torch.flip(seg, dims=[-1])
        return img, seg, bbox
    
    def load_imgs(self, img_path, bbox):
        img = Image.open(img_path).convert('RGB')
        width, height = img.size

        if bbox is not None:
            r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - r)
            y2 = np.minimum(height, center_y + r)
            x1 = np.maximum(0, center_x - r)
            x2 = np.minimum(width, center_x + r)

        cimg = img.crop([x1, y1, x2, y2])
        return self.transform(cimg)        

    def load_segs(self, seg_path, bbox):
        img = Image.open(seg_path).convert('1')
        width, height = img.size

        if bbox is not None:
            r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - r)
            y2 = np.minimum(height, center_y + r)
            x1 = np.maximum(0, center_x - r)
            x2 = np.minimum(width, center_x + r)

        cimg = img.crop([x1, y1, x2, y2])
        return self.transform_seg(cimg)

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True,
                shuffle=False
            )

            for item in sample_loader:
                yield item


class BirdsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        self.data_root = args.data_root
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.train_dataset = CubDataset(
            args.data_root, args.resolution, 'train', 
            use_flip=False, use_rescale=args.use_rescale
        )
        self.val_dataset = CubDataset(
            args.data_root, args.resolution, 'val', 
            use_flip=False, use_rescale=args.use_rescale
        )
        self.test_dataset = CubDataset(
            args.data_root, args.resolution, 'test', 
            use_flip=False, use_rescale=args.use_rescale
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

'''test'''
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.data_root = '/scratch/generalvision/Birds'
    args.use_rescale = False
    args.batch_size = 40
    args.num_workers = 4
    args.resolution = 128, 128

    datamodule = BirdsDataModule(args)
    perm = torch.arange(16)
    idx = perm[: 8]
    dl = datamodule.train_dataloader()
    it = iter(dl)
    batch = next(it)
    batch_img, batch_masks = batch['image'], batch['mask']
    # print(batch_img.shape, batch_masks.shape)
    # print(batch_masks[0, 0, :32, :32])

