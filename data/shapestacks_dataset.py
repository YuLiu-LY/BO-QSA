import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

import torch
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl

from PIL import Image 
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from methods.utils import rescale
from third_party.shapestacks.shapestacks_provider import _get_filenames_with_labels
from third_party.shapestacks.segmentation_utils import load_segmap_as_matrix


MAX_SHAPES = 6
CENTRE_CROP = 196


class ShapeStacksDataset(Dataset):
    def __init__(
        self, data_dir, split_name, mode, img_size=128,
        load_instances=True, shuffle_files=False, use_rescale=True
        ):
        self.data_dir = data_dir
        self.img_size = img_size
        self.load_instances = load_instances

        # Files
        split_dir = os.path.join(data_dir, 'splits', split_name)
        filenames, self.stability_labels = _get_filenames_with_labels(
            mode, data_dir, split_dir)
        self.filenames = filenames

        # Shuffle files?
        if shuffle_files:
            # print(f"Shuffling {len(self.filenames)} files")
            idx = np.arange(len(self.filenames), dtype='int32')
            np.random.shuffle(idx)
            self.filenames = [self.filenames[i] for i in list(idx)]
            self.stability_labels = [self.stability_labels[i] for i in list(idx)]

        # Transforms
        T = [transforms.CenterCrop(CENTRE_CROP)]
        if img_size != CENTRE_CROP:
            T.append(transforms.Resize(img_size))
        T.append(transforms.ToTensor())
        if use_rescale:
            T.append(transforms.Lambda(rescale))
        self.transform = transforms.Compose(T)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # --- Load image ---
        # File name example:
        # data_dir + /recordings/env_ccs-hard-h=2-vcom=0-vpsf=0-v=60/
        # rgb-w=5-f=2-l=1-c=unique-cam_7-mono-0.png
        file = self.filenames[idx]
        img = Image.open(file)
        output = {'image': self.transform(img)}

        # --- Load instances ---
        
        if self.load_instances:
            file_split = file.split('/')
            # cam = file_split[4].split('-')[5][4:]
            # map_path = os.path.join(
            #     self.data_dir, 'iseg', file_split[3],
            #     'iseg-w=0-f=0-l=0-c=original-cam_' + cam + '-mono-0.map')
            cam = file_split[-1].split('-')[5][4:]
            map_path = os.path.join(
                self.data_dir, 'recordings', file_split[-2],
                'iseg-w=0-f=0-l=0-c=original-cam_' + cam + '-mono-0.map')
            masks = load_segmap_as_matrix(map_path)
            masks = np.expand_dims(masks, 0)
            masks = np_img_centre_crop(masks, CENTRE_CROP)
            masks = torch.FloatTensor(masks)
            if self.img_size != masks.shape[2]:
                masks = masks.unsqueeze(0)
                masks = F.interpolate(masks, size=self.img_size)
                masks = masks.squeeze(0)
            masks = (masks * 255).int().view(1, self.img_size, self.img_size)
            output['mask'] = masks
        return output


class ShapeStacksDataModule(pl.LightningDataModule):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        self.data_root = args.data_root 
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.train_dataset = ShapeStacksDataset(
            self.data_root, 'default', 'train', 
            args.resolution[0], use_rescale=args.use_rescale
        )
        self.val_dataset = ShapeStacksDataset(
            self.data_root, 'default', 'eval', 
            args.resolution[0], use_rescale=args.use_rescale
        )
        self.test_dataset = ShapeStacksDataset(
            self.data_root, 'default', 'test', 
            args.resolution[0], use_rescale=args.use_rescale
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
            shuffle=True,
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


def np_img_centre_crop(np_img, crop_dim, batch=False):
    # np_img: [c, dim1, dim2] if batch == False else [batch_sz, c, dim1, dim2]
    shape = np_img.shape
    if batch:
        s2 = (shape[2]-crop_dim)//2
        s3 = (shape[3]-crop_dim)//2
        return np_img[:, :, s2:s2+crop_dim, s3:s3+crop_dim]
    else:
        s1 = (shape[1]-crop_dim)//2
        s2 = (shape[2]-crop_dim)//2
        return np_img[:, s1:s1+crop_dim, s2:s2+crop_dim]

'''test'''
if __name__ == '__main__':
    import argparse
    import torch
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.data_root = '/scratch/generalvision/ShapeStacks'
    args.use_rescale = False
    args.batch_size = 128
    args.num_workers = 4
    args.resolution = 128, 128

    datamodule = ShapeStacksDataModule(args)
    dl = datamodule.val_dataloader()
    batch = next(iter(dl))
    batch_img, batch_masks = batch['image'], batch['mask']
    batch_path = batch['mask_path']
    print(batch_img.shape, batch_masks.shape)
