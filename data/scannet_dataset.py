import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

from PIL import Image
from glob import glob
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from methods.utils import rescale


class ScanNetDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split_name: str,
        split:str,
        use_rescale=True,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.split_name = split_name

        T = [
                transforms.ToTensor(),
            ]
        if use_rescale:
            T.append(transforms.Lambda(rescale))  # rescale between -1 and 1
        self.transform_img = transforms.Compose(T)
        self.transform_mask = transforms.ToTensor()
        self.img_files = self.get_files()
        
    def __getitem__(self, index: int):
        img_path = self.img_files[index]
        img = Image.open(img_path)
        img = img.convert("RGB")
        image = self.transform_img(img)

        if self.split != 'train':
            mask_path = img_path.replace('image', 'mask')
            mask = Image.open(mask_path)
            mask = (self.transform_mask(mask) * 255).int()
            return {
                'image': image,
                'mask': mask,
            }
        else:
            return {
                'image': image,
            }

    def __len__(self):
        return len(self.img_files)
    
    def get_files(self):
        img_files = sorted(glob(f'{self.data_root}/{self.split}/{self.split_name}/*.png'))
        return img_files
              

class ScanNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.train_dataset = ScanNetDataset(args.data_root, args.split_name, 'train', args.use_rescale)
        self.val_dataset = ScanNetDataset(args.data_root, args.split_name, 'test', args.use_rescale)
        self.test_dataset = ScanNetDataset(args.data_root, args.split_name, 'test', args.use_rescale)

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
    # args.data_root = '/scratch/generalvision/ObjectsRoom'
    # args.use_rescale = False
    # args.batch_size = 20
    # args.num_workers = 4

    # datamodule = ObjectsRoomDataModule(args)
    # dl = datamodule.val_dataloader()
    # it = iter(dl)
    # batch = next(it)
    # batch_img, batch_masks = batch['image'], batch['mask']
    # print(batch_img.shape, batch_masks.shape)
    # for mask in batch_masks:
    #     print(mask.unique())
    # print(batch_masks[0, 0, 16:48, 16:32])

    
