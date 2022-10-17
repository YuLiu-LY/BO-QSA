import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

from typing import Tuple
import numpy as np
import torch
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import cv2

from methods.utils import rescale


class CarsDataset(Dataset):
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

        self.file_meta = self.collect_meta()

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
        sel_indices_tr = np.load('{}/data_mrcnn_train_select.npy'.format(self.ROOT_DIR))
        sel_indices_te = np.load('{}/data_mrcnn_test_select.npy'.format(self.ROOT_DIR))
        
        if self.data_split == 'train': # training split
            filenames = ['data_mrcnn/train/resized/{}'.format(token) for token in sel_indices_tr]
        elif self.data_split == 'val': # validation split
            filenames = ['data_mrcnn/train/resized/{}'.format(token) for token in sel_indices_tr]
        elif self.data_split == 'test': # testing split
            filenames = ['data_mrcnn/test/resized/{}'.format(token) for token in sel_indices_te]
        return filenames

    def load_item(self, index):
        key = self.file_meta[index]        
        
        data_dir = self.ROOT_DIR            

        img_path = '%s/%s_resized.png' % (data_dir, key)
        img = self.load_imgs(img_path)

        seg_path = '%s/%s_maskresized.png' % (data_dir, key)        
        seg = self.load_segs(seg_path)

        if self.use_flip and np.random.uniform() > 0.5:
            img = torch.flip(img, dims=[-1])
            seg = torch.flip(seg, dims=[-1])  
        return img, seg, index
    
    def load_imgs(self, img_path):
        img = cv2.imread(img_path)
        img = Image.fromarray(img)

        return self.transform(img)        

    def load_segs(self, seg_path):
        img = Image.open(seg_path).convert('1')        

        return self.transform_seg(img)        

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


class CarsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        self.data_root = args.data_root
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.train_dataset = CarsDataset(
            args.data_root, args.resolution, 'train', 
            use_flip=False, use_rescale=args.use_rescale
        )
        self.val_dataset = CarsDataset(
            args.data_root, args.resolution, 'val', 
            use_flip=False, use_rescale=args.use_rescale
        )
        self.test_dataset = CarsDataset(
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
# if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    # args.data_root = '/scratch/generalvision/Cars'
    # args.use_rescale = False
    # args.batch_size = 40
    # args.num_workers = 4
    # args.resolution = 128, 128

    # datamodule = CarsDataModule(args)
    # perm = torch.arange(16)
    # idx = perm[: 8]
    # dl = datamodule.train_dataloader()
    # batch = next(iter(dl))
    # batch_img, batch_masks = batch['image'], batch['mask']
    # print(batch_img.shape, batch_masks.shape)
    # print(batch_masks[0, 0, :32, :32])