from typing import Tuple
import json
import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

import torch
import pytorch_lightning as pl
from PIL import Image
from glob import glob
from tqdm import tqdm
from pycocotools.mask import decode
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from methods.utils import rescale


class PTRDataset(Dataset):
    def __init__(
        self,
        resolution: Tuple[int, int],
        data_root: str,
        max_n_objects: int,
        split: str = "train",
        use_rescale=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.data_root = data_root
        self.max_n_objects = max_n_objects
        self.split = split

        crop_size = [400, 700]
        self.transform_seg = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Resize(resolution),
        ])
        trans = [
            transforms.CenterCrop(crop_size),
            transforms.Resize(resolution),            
            transforms.ToTensor(),
        ]
        if use_rescale:
            trans.append(transforms.Lambda(rescale))
        self.transform = transforms.Compose(trans)
        self.scene_files, self.img_files = self.get_files()

    def __getitem__(self, index: int):
        img_path = self.img_files[index]

        img = Image.open(img_path)
        img = img.convert("RGB")
        image = self.transform(img)

        if self.split != 'train':
            scene_path = self.scene_files[index]
            scene = json.load(open(scene_path))
            num_objects_in_scene = len(scene["objects"])
            masks = []
            for i in range(num_objects_in_scene):
                mask_obj = decode(scene['objects'][i]["obj_mask"])
                mask_obj = self.transform_seg(torch.from_numpy(mask_obj).unsqueeze(0))
                masks.append(mask_obj)
            mask_bg = 1 - sum(masks)
            # add background mask
            masks.insert(0, mask_bg)
            masks = torch.stack(masks, dim=0)
            masks = torch.argmax(masks, dim=0)
            return {
                'image': image,
                'mask': masks,
            }
        else:
            return {
                'image': image,
            }


    def __len__(self):
        return len(self.scene_files)

    def get_files(self):
        scene_paths_raw = glob(f'{self.data_root}/{self.split}_scenes/PTR_{self.split}_*.json')
        scene_paths = []
        img_paths = []
        print(f'finding {self.split} scenes...')
        for item in tqdm(scene_paths_raw):
            scene_path = f'{item}'
            with open(scene_path) as f:
                scene = json.load(f)
                num_objects_in_scene = len(scene["objects"])
                if num_objects_in_scene <= self.max_n_objects:
                    image_path = os.path.join(f'{self.data_root}/{self.split}_images', scene['image_filename'])
                    if os.path.exists(image_path):
                        scene_paths.append(scene_path)
                        img_paths.append(image_path)
                f.close()
        print(f'{self.split}: {len(scene_paths)} images')
        return scene_paths, img_paths
    
    
class PTRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        self.data_root = args.data_root 
        self.batch_size = args.batch_size
        self.max_n_objects = 6
        self.num_workers = args.num_workers

        self.train_dataset = PTRDataset(
            resolution=args.resolution,
            data_root=self.data_root,
            split="train",
            max_n_objects=self.max_n_objects,
            use_rescale=args.use_rescale
        )
        self.val_dataset = PTRDataset(
            resolution=args.resolution,
            data_root=self.data_root,
            split="val",
            max_n_objects=self.max_n_objects,
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
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
'''test'''
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     args = parser.parse_args()
#     args.data_root = '/scratch/generalvision/PTR'
#     args.use_rescale = False
#     args.batch_size = 40
#     args.num_workers = 4
#     args.resolution = 128, 128

#     datamodule = PTRDataModule(args)
#     perm = torch.arange(16)
#     idx = perm[: 8]
#     dl = datamodule.val_dataloader()
#     batch = next(iter(dl))
#     batch_img, batch_masks = batch['image'], batch['mask']
#     print(batch_img.shape, batch_masks.shape)
#     print(batch_masks[0, 0, 32:64, 32:64])
