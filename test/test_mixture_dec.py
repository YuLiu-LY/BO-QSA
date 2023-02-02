from glob import glob
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor

import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

from models.model_mixture_dec import SlotAttentionModel
from methods.method_mixture_dec import SlotAttentionMethod
from methods.utils import ImageLogCallback, set_random_seed, state_dict_ckpt

import tensorboard
import json
import argparse

from data.birds_dataset import BirdsDataModule
from data.shapestacks_dataset import ShapeStacksDataModule
from data.flowers_dataset import FlowersDataModule
from data.ptr_dataset import PTRDataModule
from data.clevrtex_dataset import CLEVRTEXDataModule
from data.dogs_dataset import DogsDataModule
from data.cars_dataset import CarsDataModule
from data.ycb_dataset import YCBDataModule
from data.scannet_dataset import ScanNetDataModule
from data.coco_dataset import COCODataModule
from data.objectsroom_dataset import ObjectsRoomDataModule


datamodules = {
    'shapestacks': ShapeStacksDataModule,
    'birds': BirdsDataModule,
    'dogs': DogsDataModule,
    'cars': CarsDataModule,
    'clevrtex': CLEVRTEXDataModule,
    'ptr': PTRDataModule,
    'flowers': FlowersDataModule,
    'objectsroom': ObjectsRoomDataModule,
    'ycb': YCBDataModule,
    'scannet': ScanNetDataModule,
    'coco': COCODataModule,
}

monitors = {
    'iou': 'avg_IoU',
    'ari': 'avg_ARI_FG',
    'ap': 'avg_AP@05',
}

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='')
parser.add_argument('--data_root', default='')
parser.add_argument('--split_name', type=str, default='image', help='split for YCB, COCO, and ScanNet; for CLEVRTEX is full, outd, and camo')
parser.add_argument('--log_name', default='test')
parser.add_argument('--log_path', default='../../results/')
parser.add_argument('--ckpt_path', default='ckpt.pt.tar')
parser.add_argument('--test_ckpt_path', default='.ckpt')

parser.add_argument('--evaluate', type=str, default='ari', help='ari or iou')
parser.add_argument('--monitor', type=str, default='avg_ARI_FG', help='avg_ARI_FG or avg_IoU')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--max_steps', type=int, default=250000)
parser.add_argument('--max_epochs', type=int, default=100000)
parser.add_argument('--num_sanity_val_steps', type=int, default=1)
parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
parser.add_argument('--n_samples', type=int, default=16)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--gpus', type=int, default=0)

parser.add_argument('--grad_clip', type=float, default=1.0)
parser.add_argument('--resolution', type=int, nargs='+', default=[128, 128])
parser.add_argument('--init_resolution', type=int, nargs='+', default=[8, 8])
parser.add_argument('--encoder_channels', type=int, nargs='+', default=[64, 64, 64, 64])
parser.add_argument('--decoder_channels', type=int, nargs='+', default=[64, 64, 64, 64])
parser.add_argument('--encoder_strides', type=int, nargs='+', default=[2, 1, 1, 1])
parser.add_argument('--decoder_strides', type=int, nargs='+', default=[2, 2, 2, 2])
parser.add_argument('--encoder_kernel_size', type=int, default=5)
parser.add_argument('--decoder_kernel_size', type=int, default=5)

parser.add_argument('--is_logger_enabled', default=False, action='store_true')
parser.add_argument('--load_from_ckpt', default=False, action='store_true')
parser.add_argument('--use_rescale', default=False, action='store_true')
parser.add_argument('--truncate',  type=str, default='bi-level', help='bi-level or fixed-point or w/o')

parser.add_argument('--lr_sa', type=float, default=4e-4)
parser.add_argument('--warmup_steps', type=int, default=5000)
parser.add_argument('--decay_steps', type=int, default=50000)

parser.add_argument('--num_iter', type=int, default=3)
parser.add_argument('--num_slots', type=int, default=2)
parser.add_argument('--init_size', type=int, default=64)
parser.add_argument('--slot_size', type=int, default=64)
parser.add_argument('--mlp_size', type=int, default=128)
parser.add_argument('--init_method', default='embedding', help='shared_gaussian, embedding')

parser.add_argument('--sigma_steps', type=int, default=30000)
parser.add_argument('--sigma_final', type=float, default=0)
parser.add_argument('--sigma_start', type=float, default=1)


def main(args):
    set_random_seed(args.seed)
    args.monitor = monitors[args.evaluate]
    datamodule = datamodules[args.dataset](args)
    args.test_num_slots = args.num_slots
    model = SlotAttentionModel(args)
    ckpt = state_dict_ckpt(args.test_ckpt_path)
    model.load_state_dict(ckpt)
    method = SlotAttentionMethod(model=model, datamodule=datamodule, args=args)

    logger_name = 'test_' + args.log_name
    logger = pl_loggers.TensorBoardLogger(args.log_path, name=logger_name)

    trainer = Trainer(
        logger=logger if args.is_logger_enabled else False,
        default_root_dir=args.log_path,
        accelerator=None,
        gpus=args.gpus,
        callbacks=[LearningRateMonitor("step"), ImageLogCallback(),] if args.is_logger_enabled else [],
    )
    print(logger_name)
    trainer.test(method)

    # visualize
    # from torchvision import transforms
    # for i in range(2):
    #     method.eval()
    #     images = method.sample_images()
    #     images = transforms.ToPILImage()(images)
    #     images.save(f'{args.log_path}/{logger_name}/{i}.png')

if __name__ == "__main__":
    args = parser.parse_args()
    # args.test_ckpt_path = '/home/liuyu/scratch/SlotAttention/results/ptr/emb_30000_mix_seed2/version_0/checkpoints/last.ckpt'
    # args.dataset = 'objectsroom'
    # args.data_root = data_paths[args.dataset]
    # args.log_path += args.dataset
    # args.resolution = [64, 64]
    # args.encoder_strides = [1, 1, 1, 1]
    # args.decoder_strides = [2, 2, 2, 1]
    # args.num_slots = 5
    # args.is_logger_enabled = False
    # args.use_rescale = True
    paths = json.load(open('./path.json', 'r'))
    data_paths = paths['data_paths']
    args.log_path = paths['log_path']
    args.data_root = data_paths[args.dataset]
    args.log_path += args.dataset
    args.test_ckpt_path = f'./checkpoints/{args.dataset}/{args.dataset}_mix.ckpt'
    main(args)



