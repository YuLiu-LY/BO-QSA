import torch
import pytorch_lightning as pl
from torch import optim
from torchvision import utils as vutils
import torch.nn.functional as F
import math

import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

from models.model_mixture_dec import SlotAttentionModel
from methods.utils import to_rgb_from_tensor, average_ari, iou_and_dice, average_segcover
from methods.seg_metrics import Segmentation_Metrics_Calculator


class SlotAttentionMethod(pl.LightningModule):
    def __init__(self, model: SlotAttentionModel, datamodule: pl.LightningDataModule, args):
        super().__init__()
        self.model = model
        self.datamodule = datamodule
        self.args = args
        self.val_iter = iter(self.datamodule.val_dataloader())
        self.empty_cache = True
        self.evaluate = args.evaluate
        self.sigma = 0
        self.sample_num = 0
        if self.evaluate == "ap":
            self.seg_metric_logger = Segmentation_Metrics_Calculator(max_ins_num=self.args.num_slots)

    def forward(self, input, **kwargs):
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):
        batch_img = batch['image']
        self.sigma = self.cosine_anneal(self.global_step, self.args.sigma_steps, start_value=self.args.sigma_start, final_value=self.args.sigma_final)
        out = self.model.forward(batch_img, sigma=self.sigma)
        logs = {}
        logs['sigma'] = self.sigma
        loss = out['mse']
        logs['loss'] = loss.item()
        self.log_dict(logs, sync_dist=True)
        return loss

    def sample_images(self):
        if self.sample_num % (len(self.val_iter) - 1) == 0:
            self.val_iter = iter(self.datamodule.val_dataloader())
        self.sample_num += 1

        batch = next(self.val_iter)
        batch_img = batch['image'][:self.args.n_samples]
        mask_gt = batch['mask'][:self.args.n_samples]
        if self.args.gpus > 0:
            batch_img = batch_img.to(self.device)

        out = self.model.forward(batch_img, self.sigma)
        recon, masks = out['recon'], out['masks']
        recons = out['recons']

        if self.args.use_rescale:
            out = to_rgb_from_tensor(
                torch.cat(
                    [
                        batch_img.unsqueeze(1),  # original images
                        recon.unsqueeze(1),  # reconstructions
                        recons * masks + 1 - masks,
                    ],
                    dim=1,
                )
            ).cpu()
        else:
            out = torch.cat(
                    [
                        batch_img.unsqueeze(1),  # original images
                        recon.unsqueeze(1),  # reconstructions
                        recons * masks + 1 - masks,
                    ],
                    dim=1,
                ).cpu()
        # visualize the masks
        m = (1 - masks).expand(-1, -1, 3, -1, -1).cpu()
        out = torch.cat([out, m], dim=1) # add masks
        out = torch.cat([out, mask_gt.unsqueeze(1).expand(-1, -1, 3, -1, -1).cpu()], dim=1) # add gt masks

        B, C, H, W = batch_img.shape
        images = vutils.make_grid(
            out.reshape(out.shape[0] * out.shape[1], C, H, W).cpu(), normalize=False, nrow=out.shape[1],
            padding=3, pad_value=0
        )

        return images

    def validation_step(self, batch, batch_idx):
        if self.empty_cache:
            torch.cuda.empty_cache()
            self.empty_cache = False

        batch_img = batch['image']
        masks_gt = batch['mask']
        out = self.model.forward(batch_img, self.sigma)
        mse, masks = out['mse'], out['masks'] 
        # mask shape: B, K, 1, H, W
        output = {
            'mse': mse,
        }
        if self.evaluate == 'ari':
            m = masks.detach().argmax(dim=1)
            ari, _ = average_ari(m, masks_gt)
            ari_fg, _ = average_ari(m, masks_gt, True)
            msc_fg, _ = average_segcover(masks_gt, m, True)
            output['ARI'] = ari.to(self.device)
            output['ARI_FG'] = ari_fg.to(self.device)
            output['MSC_FG'] = msc_fg.to(self.device)
        elif self.evaluate == 'iou':
            K = self.args.num_slots
            m = F.one_hot(masks.argmax(dim=1), K).permute(0, 4, 1, 2, 3)
            iou, dice = iou_and_dice(m[:, 0], masks_gt)
            for i in range(1, K):
                iou1, dice1 = iou_and_dice(m[:, i], masks_gt)
                iou = torch.max(iou, iou1)
                dice = torch.max(dice, dice1)
            output['IoU'] = iou.mean()
            output['Dice'] = dice.mean()
        elif self.evaluate == "ap":
            masks = masks.detach().cpu()[:, :, 0, :, :]           # (B, K, 1, H, W) -> (B, K, H, W)
            pred_mask_conf, _ = masks.max(dim=1)
            confidence_mask = (pred_mask_conf > 0.5)
            pred_mask_oh = masks.argmax(dim=1)                    # (B, K, H, W) -> (B, H, W)
            gt_mask_oh = masks_gt[:, 0, :, :].cpu()               # (B, 1, H, W) -> (B, H, W)                   
            gt_fg_batch = (gt_mask_oh != 0)
            self.seg_metric_logger.update_new_batch(
                pred_mask_batch=pred_mask_oh,
                gt_mask_batch=gt_mask_oh,
                valid_pred_batch=confidence_mask,
                gt_fg_batch=gt_fg_batch,
                pred_conf_mask_batch=pred_mask_conf
            )
        return output

    def validation_epoch_end(self, outputs):
        self.empty_cache = True
        if self.evaluate != "ap":
            keys = outputs[0].keys()
            logs = {}
            for k in keys:
                v = torch.stack([x[k] for x in outputs]).mean()
                logs['avg_' + k] = v
            self.log_dict(logs, sync_dist=True)
            print("; ".join([f"{k}: {v.item():.6f}" for k, v in logs.items()]))
        else:
            outputs = self.seg_metric_logger.calculate_score_summary()
            keys = outputs.keys()
            logs = {}
            for k in keys:
                if k not in ['AP@05','PQ','F1','precision','recall']:
                    continue
                v = outputs[k]
                logs['avg_' + k] = v
            self.log_dict(logs, sync_dist=True)
            print("; ".join([f"{k}: {v:.6f}" for k, v in logs.items()]))
            self.seg_metric_logger.reset()
    
    def test_step(self, batch, batch_idx):
        if self.empty_cache:
            torch.cuda.empty_cache()
            self.empty_cache = False

        batch_img = batch['image']
        masks_gt = batch['mask']
        out = self.model.forward(batch_img, sigma=0)
        mse, masks = out['mse'], out['masks'] 
        # mask shape: B, K, 1, H, W
        output = {
            'mse': mse * self.args.resolution[0] * self.args.resolution[1],
        }
        if self.evaluate == 'ari':
            m = masks.detach().argmax(dim=1)
            ari, _ = average_ari(m, masks_gt)
            ari_fg, _ = average_ari(m, masks_gt, True)
            msc_fg, _ = average_segcover(masks_gt, m, True)
            output['ARI'] = ari.to(self.device)
            output['ARI_FG'] = ari_fg.to(self.device)
            output['MSC_FG'] = msc_fg.to(self.device)
        elif self.evaluate == 'iou':
            K = self.args.num_slots
            m = F.one_hot(masks.argmax(dim=1), K).permute(0, 4, 1, 2, 3)
            iou, dice = iou_and_dice(m[:, 0], masks_gt)
            for i in range(1, K):
                iou1, dice1 = iou_and_dice(m[:, i], masks_gt)
                iou = torch.max(iou, iou1)
                dice = torch.max(dice, dice1)
            output['IoU'] = iou.mean()
            output['Dice'] = dice.mean()
        elif self.evaluate == "ap":
            masks = masks.detach().cpu()[:, :, 0, :, :]           # (B, K, 1, H, W) -> (B, K, H, W)
            pred_mask_conf, _ = masks.max(dim=1)
            confidence_mask = (pred_mask_conf > 0.5)
            pred_mask_oh = masks.argmax(dim=1)                    # (B, K, H, W) -> (B, H, W)
            gt_mask_oh = masks_gt[:, 0, :, :].cpu()               # (B, 1, H, W) -> (B, H, W)                   
            gt_fg_batch = (gt_mask_oh != 0)
            self.seg_metric_logger.update_new_batch(
                pred_mask_batch=pred_mask_oh,
                gt_mask_batch=gt_mask_oh,
                valid_pred_batch=confidence_mask,
                gt_fg_batch=gt_fg_batch,
                pred_conf_mask_batch=pred_mask_conf
            )
        return output
        

    def test_epoch_end(self, outputs):
        self.empty_cache = True
        if self.evaluate != "ap":
            keys = outputs[0].keys()
            logs = {}
            for k in keys:
                v = torch.stack([x[k] for x in outputs]).mean()
                logs['avg_' + k] = v
            self.log_dict(logs, sync_dist=True)
            print("; ".join([f"{k}: {v.item():.6f}" for k, v in logs.items()]))
        else:
            outputs = self.seg_metric_logger.calculate_score_summary()
            keys = outputs.keys()
            logs = {}
            for k in keys:
                if k not in ['AP@05','PQ','F1','precision','recall']:
                    continue
                v = outputs[k]
                logs['avg_' + k] = v
            self.log_dict(logs, sync_dist=True)
            print("; ".join([f"{k}: {v:.6f}" for k, v in logs.items()]))
            self.seg_metric_logger.reset()
    def configure_optimizers(self):

        warmup_steps = self.args.warmup_steps
        decay_steps = self.args.decay_steps

        def lr_scheduler_warm(step: int):
            if step < warmup_steps:
                factor = step / warmup_steps
            else:
                factor = 1
            factor *= 0.5 ** (step / decay_steps)
            return factor

        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr_sa)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_scheduler_warm)

        return (
            [optimizer],
            [{"scheduler": scheduler, "interval": "step",}],
        )

    def cosine_anneal(self, step, final_step, start_step=0, start_value=1.0, final_value=0.1):
    
        assert start_value >= final_value
        assert start_step <= final_step
        
        if step < start_step:
            value = start_value
        elif step >= final_step:
            value = final_value
        else:
            a = 0.5 * (start_value - final_value)
            b = 0.5 * (start_value + final_value)
            progress = (step - start_step) / (final_step - start_step)
            value = a * math.cos(math.pi * progress) + b
        return value
