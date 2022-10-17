import os
import sys
import json
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

import torch
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as Ft
from PIL import Image

from methods.utils import rescale

class DatasetReadError(ValueError):
    pass


class CLEVRTEX(Dataset):
    ccrop_frac = 0.8
    splits = {
        'test': (0., 0.1),
        'val': (0.1, 0.2),
        'train': (0.2, 1.)
    }
    shape = (3, 240, 320)
    variants = {'full', 'pbg', 'vbg', 'grassbg', 'camo', 'outd'}

    def _index_with_bias_and_limit(self, idx):
        if idx >= 0:
            idx += self.bias
            if idx >= self.limit:
                raise IndexError()
        else:
            idx = self.limit + idx
            if idx < self.bias:
                raise IndexError()
        return idx

    def _reindex(self):
        print(f'Indexing {self.basepath}')

        img_index = {}
        msk_index = {}
        met_index = {}

        prefix = f"CLEVRTEX_{self.dataset_variant}_"

        img_suffix = ".png"
        msk_suffix = "_flat.png"
        met_suffix = ".json"

        _max = 0
        for img_path in self.basepath.glob(f'**/{prefix}??????{img_suffix}'):
            indstr = img_path.name.replace(prefix, '').replace(img_suffix, '')
            msk_path = img_path.parent / f"{prefix}{indstr}{msk_suffix}"
            met_path = img_path.parent / f"{prefix}{indstr}{met_suffix}"
            indstr_stripped = indstr.lstrip('0')
            if indstr_stripped:
                ind = int(indstr)
            else:
                ind = 0
            if ind > _max:
                _max = ind

            if not msk_path.exists():
                raise DatasetReadError(f"Missing {msk_suffix.name}")

            if ind in img_index:
                raise DatasetReadError(f"Duplica {ind}")

            img_index[ind] = img_path
            msk_index[ind] = msk_path
            if self.return_metadata:
                if not met_path.exists():
                    raise DatasetReadError(f"Missing {met_path.name}")
                met_index[ind] = met_path
            else:
                met_index[ind] = None

        if len(img_index) == 0:
            raise DatasetReadError(f"No values found")
        missing = [i for i in range(0, _max) if i not in img_index]
        if missing:
            raise DatasetReadError(f"Missing images numbers {missing}")

        return img_index, msk_index, met_index

    def _variant_subfolder(self):
        return f"clevrtex_{self.dataset_variant.lower()}"

    def __init__(self,
                 path,
                 dataset_variant='full',
                 split='train',
                 use_rescale=True,
                 crop=True,
                 resize=(128, 128),
                 return_metadata=False):
        path = Path(path)
        self.return_metadata = return_metadata
        self.crop = crop
        self.resize = resize
        self.rescale = rescale if use_rescale else None

        if dataset_variant not in self.variants:
            raise DatasetReadError(f"Unknown variant {dataset_variant}; [{', '.join(self.variants)}] available ")

        if split not in self.splits:
            raise DatasetReadError(f"Unknown split {split}; [{', '.join(self.splits)}] available ")
        if dataset_variant == 'outd':
            # No dataset splits in
            split = None

        self.dataset_variant = dataset_variant
        self.split = split

        self.basepath = Path(path)
        if not self.basepath.exists():
            raise DatasetReadError()
        sub_fold = self._variant_subfolder()
        if self.basepath.name != sub_fold:
            self.basepath = self.basepath / sub_fold
        #         try:
        #             with (self.basepath / 'manifest_ind.json').open('r') as inf:
        #                 self.index = json.load(inf)
        #         except (json.JSONDecodeError, IOError, FileNotFoundError):
        self.index, self.mask_index, self.metadata_index = self._reindex()

        print(f"Sourced {dataset_variant} ({split}) from {self.basepath}")

        bias, limit = self.splits.get(split, (0., 1.))
        if isinstance(bias, float):
            bias = int(bias * len(self.index))
        if isinstance(limit, float):
            limit = int(limit * len(self.index))
        self.limit = limit
        self.bias = bias

    def _format_metadata(self, meta):
        """
        Drop unimportanat, unsued or incorrect data from metadata.
        Data may become incorrect due to transformations,
        such as cropping and resizing would make pixel coordinates incorrect.
        Furthermore, only VBG dataset has color assigned to objects, we delete the value for others.
        """
        objs = []
        for obj in meta['objects']:
            o = {
                'material': obj['material'],
                'shape': obj['shape'],
                'size': obj['size'],
                'rotation': obj['rotation'],
            }
            if self.dataset_variant == 'vbg':
                o['color'] = obj['color']
            objs.append(o)
        return {
            'ground_material': meta['ground_material'],
            'objects': objs
        }

    def __len__(self):
        return self.limit - self.bias

    def __getitem__(self, ind):
        ind = self._index_with_bias_and_limit(ind)

        img = Image.open(self.index[ind])
        msk = Image.open(self.mask_index[ind])

        if self.crop:
            crop_size = int(0.8 * float(min(img.width, img.height)))
            img = img.crop(((img.width - crop_size) // 2,
                            (img.height - crop_size) // 2,
                            (img.width + crop_size) // 2,
                            (img.height + crop_size) // 2))
            msk = msk.crop(((msk.width - crop_size) // 2,
                            (msk.height - crop_size) // 2,
                            (msk.width + crop_size) // 2,
                            (msk.height + crop_size) // 2))
        if self.resize:
            img = img.resize(self.resize, resample=Image.BILINEAR)
            msk = msk.resize(self.resize, resample=Image.NEAREST)

        img = Ft.to_tensor(np.array(img)[..., :3])
        if self.rescale is not None:
            img = self.rescale(img)
        msk = torch.from_numpy(np.array(msk))[None]

        ret = (ind, img, msk)

        if self.return_metadata:
            with self.metadata_index[ind].open('r') as inf:
                meta = json.load(inf)
            ret = (ind, img, msk, self._format_metadata(meta))
        
        return {
            "image": img,
            "mask": msk
        }


class CLEVRTEXDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_root = args.data_root
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        args.split_name = "full"
        args.use_crop = False

        self.train_dataset = CLEVRTEX(self.data_root, args.split_name, "train", args.use_rescale, 
                                        resize=(args.resolution[0], args.resolution[1]), crop=args.use_crop)
        self.val_dataset = CLEVRTEX(self.data_root, args.split_name, "val", args.use_rescale,
                                        resize=(args.resolution[0], args.resolution[1]), crop=args.use_crop)
        self.test_dataset = CLEVRTEX(self.data_root, args.split_name, "test", args.use_rescale,
                                        resize=(args.resolution[0], args.resolution[1]), crop=args.use_crop)

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

'''test'''
# if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    # args.data_root = '/scratch/generalvision/CLEVRTEX'
    # args.use_rescale = False
    # args.batch_size = 40
    # args.num_workers = 4
    # args.resolution = 128, 128
    # args.max_n_objects = 6

    # datamodule = CLEVRTEXDataModule(args)
    # dl = datamodule.train_dataloader()
    # it = iter(dl)
    # batch = next(it)
    # batch_img, batch_masks = batch['image'], batch['mask']
    # print(batch_img.shape, batch_masks.shape)
    # print(batch_masks[0, 0, :32, :32])
