import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

import numpy as np
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from scipy import io

from methods.utils import rescale


class FlowersDataset(Dataset):
    def __init__(
        self, 
        data_root, 
        resolution=[128, 128], 
        data_split='train',
        use_rescale=False, # rescale to [-1, 1]
    ):
        super(FlowersDataset, self).__init__()
        self.files =  io.loadmat(os.path.join(data_root, "setid.mat"))
        if data_split == 'train':
            self.files = self.files.get('tstid')[0]
        elif data_split == 'val':
            self.files = self.files.get('valid')[0]
        else:
            self.files = self.files.get('trnid')[0]
        trans = [
            transforms.Resize(resolution),            
            transforms.ToTensor(),
        ]
        self.transform_seg = transforms.Compose(trans)
        if use_rescale:
            trans.append(transforms.Lambda(rescale))
        self.transform = transforms.Compose(trans)
        self.datapath = data_root

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        imgname = "image_%05d.jpg" % self.files[idx]
        segname = "segmim_%05d.jpg" % self.files[idx]
        img = self.transform(Image.open(os.path.join(self.datapath, "jpg", imgname)))
        seg = np.array(Image.open(os.path.join(self.datapath, "segmim", segname)))
        seg = 1 - ((seg[:,:,0:1] == 0) + (seg[:,:,1:2] == 0) + (seg[:,:,2:3] == 254))
        seg = (seg * 255).astype('uint8').repeat(3,axis=2)
        seg = self.transform_seg(Image.fromarray(seg))[:1]

        return {
            'image': img,
            'mask': seg, 
        }  


class FlowersDataModule(pl.LightningDataModule):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        self.data_root = args.data_root
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.train_dataset = FlowersDataset(
            args.data_root, args.resolution, 'train', 
            use_rescale=args.use_rescale
        )
        self.val_dataset = FlowersDataset(
            args.data_root, args.resolution, 'val', 
            use_rescale=args.use_rescale
        )
        self.test_dataset = FlowersDataset(
            args.data_root, args.resolution, 'test', 
            use_rescale=args.use_rescale
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
# if __name__ == "__main__":
    # import argparse
    # import torch
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    # args.data_root = '/scratch/generalvision/Flowers'
    # args.use_rescale = False
    # args.batch_size = 40
    # args.num_workers = 4
    # args.resolution = [128, 128]

    # datamodule = FlowersDataModule(args)
    # perm = torch.arange(16)
    # idx = perm[: 8]
    # dl = datamodule.train_dataloader()
    # batch = next(iter(dl))
    # batch_img, batch_masks = batch['image'], batch['mask']
    # print(batch_img.shape, batch_masks.shape)
    # print(batch_masks[0, 0, :32, :32])