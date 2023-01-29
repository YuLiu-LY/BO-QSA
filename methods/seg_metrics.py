import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
import json
import torch
from sklearn.metrics import adjusted_rand_score
from skimage.morphology import convex_hull_image, disk, binary_dilation, binary_erosion
from torch import nn
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
import sys


# General util function to get the boundary of a binary mask.
# @source: https://github.com/bowenc0221/boundary-iou-api/blob/master/boundary_iou/utils/boundary_utils.py
def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def convert_to_tensor(data):
    if isinstance(data, torch.Tensor):
        return data.long()
    elif isinstance(data, np.ndarray):
        data = data.astype(np.int64)
    else:
        data = data.numpy().astype(np.int64)
    data = torch.from_numpy(data)
    return data

def convert_to_float_tensor(data):
    if isinstance(data, torch.Tensor):
        return data.float()
    elif isinstance(data, np.ndarray):
        data = data.astype(np.float32)
    else:
        data = data.numpy().astype(np.float32)
    data = torch.from_numpy(data)
    return data

def convert_to_numpy(data):
    if not isinstance(data, np.ndarray):
        data = data.astype(np.int64)
    return data

def convert_to_float_numpy(data):
    if not isinstance(data, np.ndarray):
        data = data.astype(np.float32)
    return data

def calculate_iou(mask1, mask2):
    mask1_area = np.count_nonzero( mask1 )
    mask2_area = np.count_nonzero( mask2 )
    intersection = np.count_nonzero( np.logical_and( mask1, mask2 ) )
    iou = intersection/(mask1_area+mask2_area-intersection)
    return iou


'''
This class is used to compute segmentation metrics: AP@05 / PQ / F1 / Precision / Recall / SC / ARI
- MAIN FUNCTION 1: update_new_batch()
    Update class variable state with GT and prediction of from a new batch
- MAIN FUNCTION 2: calculate_score_summary()
    Calcuate segmentation metrics score with current class variable state]
'''
class Segmentation_Metrics_Calculator:
    '''
    max_ins_num: maximum possible number of objects, K
    '''
    def __init__(
        self,
        use_boundary_iou=False,
        max_ins_num=7,
    ):

        self.TP_count = 0
        self.FP_count = 0
        self.FN_count = 0
        self.gt_iou_list = []
        self.pred_iou_list = []
        self.pred_conf_list = []
        self.ari_list = []
        self.sc_per_image_list = [] ## length == number of image
        self.sc_per_dataset_list = [] ## length == number of gt objects
        self.max_ins_num = max_ins_num
        self.use_boundary_iou = use_boundary_iou

    def reset(self):
        self.TP_count = 0
        self.FP_count = 0
        self.FN_count = 0
        self.gt_iou_list = []
        self.pred_iou_list = []
        self.pred_conf_list = []
        self.ari_list = []
        self.sc_per_image_list = [] ## length == number of image
        self.sc_per_dataset_list = [] ## length == number of gt objects
    
    
    '''
    This function calculate final score given current class variable
    '''
    def calculate_score_summary(self):
        assert len(self.pred_conf_list) == len(self.pred_iou_list)
        assert self.TP_count + self.FP_count == len(self.pred_conf_list)
        assert self.TP_count + self.FN_count == len(self.gt_iou_list)
        ap_05 = self.calculate_ap(self.gt_iou_list, self.pred_iou_list, self.pred_conf_list)
        precision = self.TP_count / (self.TP_count + self.FP_count)
        recall = self.TP_count / (self.TP_count + self.FN_count)
        TP_iou_sum = 0
        for i in self.pred_iou_list:
            if i >= 0.5:
                TP_iou_sum += i
        PQ = TP_iou_sum / (self.TP_count + self.FP_count*0.5 + self.FN_count*0.5)
        F1 = self.TP_count / (self.TP_count + self.FP_count*0.5 + self.FN_count*0.5)
        assert len(self.ari_list) == len(self.sc_per_image_list) ## number of images
        assert len(self.sc_per_dataset_list) == len(self.gt_iou_list) ## number of objects
        ari = np.array(self.ari_list).mean()
        sc_per_image = np.array(self.sc_per_image_list).mean()
        sc_per_dataset = np.array(self.sc_per_dataset_list).mean()

        return {
            'AP@05': ap_05,
            'PQ': PQ.item(),
            'F1': F1.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'sc_per_image': sc_per_image,
            'sc_per_dataset': sc_per_dataset,
            'ari': ari,
            'image count': len(self.ari_list),
            'object count': len(self.sc_per_dataset_list)
        }

    '''
    This function takes a batch of prediction and ground truth, and update the following class variables accordingly;
        - TP_count
        - FP_count
        - FN_count
        - gt_iou_list
        - pred_iou_list
        - pred_conf_list
        - ari_list
        - sc_per_image_list
        - sc_per_dataset_list
    INPUT:
        - pred_mask_batch: [B, H, W]. Predicted segmentation mask
        - gt_mask_batch: [B, H, W]. GT segmentation mask
        - valid_pred_batch: [B, H, W], binary. Non-ignore area of prediction segmentation mask. 
        - gt_fg_batch: [B, H, W], binary. Foreground area of GT segmentation mask.
        - pred_conf_mask_batch: [B, H, W], each pixel is valued between 0-1. Prediction confidence mask.
    '''
    def update_new_batch(self,
            pred_mask_batch,
            gt_mask_batch,
            valid_pred_batch=None,
            gt_fg_batch=None,
            pred_conf_mask_batch=None
            ):
        assert gt_mask_batch.shape == pred_mask_batch.shape
        assert pred_conf_mask_batch.shape == pred_mask_batch.shape

        ## convert input to tensor
        pred_mask_batch = convert_to_tensor(pred_mask_batch)
        gt_mask_batch = convert_to_tensor(gt_mask_batch)
        valid_pred_batch = convert_to_tensor(valid_pred_batch) if valid_pred_batch is not None else valid_pred_batch
        gt_fg_batch = convert_to_tensor(gt_fg_batch)
        pred_conf_mask_batch = convert_to_float_tensor(pred_conf_mask_batch)
        valid_pred_batch = valid_pred_batch if valid_pred_batch is not None else torch.ones_like(pred_mask_batch)
        gt_fg_batch = gt_fg_batch if gt_fg_batch is not None else torch.ones_like(gt_mask_batch)
        bsz = gt_mask_batch.shape[0]

        for batch_idx in range(0, bsz):
            gt_mask = gt_mask_batch[batch_idx] ## [H, W]
            pred_mask = pred_mask_batch[batch_idx] ## [H, W]
            gt_fg_mask = gt_fg_batch[batch_idx] ## [H, W]
            valid_pred_mask = valid_pred_batch[batch_idx] ## [H, W]
            pred_conf_mask = pred_conf_mask_batch[batch_idx] ## [H, W]

            gt_ins_binary, pred_ins_binary, gt_ins_count, pred_ins_count, pred_conf_score = self.process_mask(
                gt_mask=gt_mask,
                pred_mask=pred_mask,
                gt_fg_mask=gt_fg_mask,
                valid_pred_mask=valid_pred_mask,
                pred_conf_mask=pred_conf_mask,
            )
            iou_matrix = self.get_iou_matrix(gt_ins_binary, pred_ins_binary)
            gt_matched_score, pred_matched_score = self.hungarian_matching(iou_matrix, gt_ins_count, pred_ins_count)
            TP = torch.sum(pred_matched_score>=0.5)
            FP = len(pred_matched_score) - torch.sum(pred_matched_score>=0.5)
            FN = len(gt_matched_score) - torch.sum(pred_matched_score>=0.5)
            self.TP_count += TP
            self.FP_count += FP
            self.FN_count += FN
            self.gt_iou_list.extend(gt_matched_score)
            self.pred_iou_list.extend(pred_matched_score)
            self.pred_conf_list.extend(pred_conf_score)
            self.sc_per_image_list.append(gt_matched_score.mean())
            self.sc_per_dataset_list.extend(gt_matched_score)
            ari = self.calculate_ari(gt_mask, pred_mask, gt_fg_mask)
            self.ari_list.append(ari)

    '''
    INPUT:
    - gt_mask: [H, W], GT segmentation mask, each pixel has an index representing segmentation id. 
    - pred_mask: [H, W], predicted segmentation mask, each pixel has an index representing predicted segmentation id. 
    - gt_fg_mask: [H, W], GT foreground mask, each pixel is 0/1 representing background/foreground.
    - valid_pred_mask: [H, W], prediction valid mask, each pixel is 0/1 representing whether to ignore this pixel at evaluation.
    - pred_conf_mask: [H, W], prediction confidence mask, each pixel values between [0,1], representing the prediction confidence for each pixel
    OUTPUT:
    - gt_ins_binary: [H, W, K], GT segmentation mask, 
                    each of K HxW represents a component, all-zero masks are paddings,
                    background component is not included.
    - pred_ins_binary: [H, W, K], predicted segmentation mask, 
                    each of HxW represents a component, all-zero masks are paddings,
                    only pixels in valid area are included, component matched with background is also excluded

    - gt_ins_count: (<=7) number of GT objects, exclude background 
    - pred_ins_count: (<=7) number of predicted objects, i.e. number of unique segment in valid area
    - pred_conf_scores: [K] confidence scores for each predicted component.
    '''
    def process_mask(self, 
            gt_mask,
            pred_mask,
            gt_fg_mask,
            valid_pred_mask,
            pred_conf_mask):

        ## try to match GT background with predicted component, 
        ## successful match when iou > 0.5
        ## if there is a match, remove that predicted component before AP clculation 
        pred_fg_mask = valid_pred_mask.clone()
        gt_bg_mask = 1-gt_fg_mask
        for pred_idx in torch.unique(torch.masked_select(pred_mask, valid_pred_mask.bool())):
            pred_idx = pred_idx.item()
            pred_ins = (pred_mask==pred_idx) * valid_pred_mask
            iou = torch.sum(pred_ins*gt_bg_mask) / (torch.sum(pred_ins) + torch.sum(gt_bg_mask) - torch.sum(pred_ins*gt_bg_mask) + 1e-6)
            if iou >= 0.5:
                pred_fg_mask *= (1-pred_ins)


        ## get unique labels from fg area
        gt_labels = torch.unique(torch.masked_select(gt_mask, gt_fg_mask.bool()))
        pred_labels = torch.unique(torch.masked_select(pred_mask, pred_fg_mask.bool()))
        gt_ins_count = len(gt_labels)
        pred_ins_count = len(pred_labels)

        ## mask bg area with a special index
        special_idx = 999
        assert special_idx not in gt_labels
        assert special_idx not in pred_labels
        gt_mask[gt_fg_mask==0] = special_idx
        pred_mask[pred_fg_mask==0] = special_idx

        ## turn into one-hot 
        gt_ins_binary = torch.zeros([gt_mask.shape[0], gt_mask.shape[1], self.max_ins_num])
        gt_ins_binary[..., :gt_ins_count] = F.one_hot(gt_mask)[..., gt_labels]
        pred_ins_binary = torch.zeros([pred_mask.shape[0], pred_mask.shape[1], self.max_ins_num])
        pred_ins_binary[..., :pred_ins_count] = F.one_hot(pred_mask)[..., pred_labels]
        pred_conf_scores = torch.zeros([pred_ins_count])
        pred_conf_list = []
        for pred_label in pred_labels:
            pred_obj_conf = (pred_conf_mask[pred_mask==pred_label]).sum() / (pred_mask==pred_label).sum()
            pred_conf_list.append(pred_obj_conf)
        pred_conf_scores[:pred_ins_count] = torch.tensor(pred_conf_list)

        if self.use_boundary_iou:
            for i in range(0, pred_ins_count):
                boundary_mask = mask_to_boundary(pred_ins_binary[..., i].clone().numpy())
                pred_ins_binary[..., i] = torch.from_numpy(boundary_mask)
            for i in range(0, gt_ins_count):
                boundary_mask = mask_to_boundary(gt_ins_binary[..., i].clone().numpy())
                gt_ins_binary[..., i] = torch.from_numpy(boundary_mask)

        ## gt_ins_binary: [H, W, max_ins_num]
        ## pred_ins_binary: [H, W, max_ins_num]
        ## pred_conf_scores: [max_ins_num]
        return gt_ins_binary, pred_ins_binary, gt_ins_count, pred_ins_count, pred_conf_scores
    
    '''
    This function calculates the IOU score between two binary masks
    INPUT:
    - gt_ins_binary: [H, W, K]
    - pred_ins_binary: [H, W, K]
    OUTPUT:
    - iou_matrix: [K, K]
    '''
    def get_iou_matrix(self, gt_ins_binary, pred_ins_binary):
        ## => [H*W, max_ins_num]
        gt_ins = gt_ins_binary.reshape((-1, self.max_ins_num)) 
        pred_ins = pred_ins_binary.reshape((-1, self.max_ins_num))

        ## => [max_ins_num, H*W]
        gt_ins = gt_ins.permute([1, 0]) 
        pred_ins = pred_ins.permute([1, 0])

        gt_ins = gt_ins[:, None, :] ## [max_ins_num, 1, H*W]
        pred_ins = pred_ins[None, :, :] ## [1, max_ins_num, H*W]

        TP = torch.sum(pred_ins * gt_ins, dim=-1) ##[max_ins_num, max_ins_num, H*W] => [max_ins_num, max_ins_num]
        FP = torch.sum(pred_ins, dim=-1) - TP
        FN = torch.sum(gt_ins, dim=-1) - TP
        iou_matrix = TP / (TP + FP + FN + 1e-6)

        ##[max_ins_num, max_ins_num]
        # row corresponding ground truth, column corresponding prediction
        return iou_matrix 
    
    
    '''
    This function finds the best match between GT and prediction to maximize IOU
    INPUT:
    - iou_matrix: [K, K]
    - gt_ins_count: (<=7) number of GT objects, exclude background 
    - pred_ins_count: (<=7) number of predicted objects, i.e. number of unique segment in valid area
    OUTPUT:
    - gt_match_score: list of length [gt_ins_count], the IOU of each GT component with its matched prediction component
    - pred_match_score: list of length [pred_ins_count], the IOU of each predicted component with its matched GT component
    '''
    def hungarian_matching(self, iou_matrix, gt_ins_count, pred_ins_count):
        '''
        iou_metrics: [max_ins_num, max_ins_num], row corresponding ground truth, column corresponding prediction
        '''
        sorted_gt_labels, matched_pred_labels = linear_sum_assignment(iou_matrix.numpy(), maximize=True)
        gt_match_score = iou_matrix[sorted_gt_labels, matched_pred_labels]
        sorted_pred_labels, matched_gt_labels = linear_sum_assignment(np.transpose(iou_matrix.numpy()), maximize=True)
        pred_matched_score = iou_matrix[matched_gt_labels, sorted_pred_labels]

        return gt_match_score[:gt_ins_count], pred_matched_score[:pred_ins_count]
    
    '''
    This function calculates ARI score
    INPUT:
    - gt_mask: [H, W], GT segmentation mask, each pixel has an index representing segmentation id. 
    - pred_mask: [H, W], predicted segmentation mask, each pixel has an index representing predicted segmentation id. 
    - gt_fg_mask: [H, W], GT foreground mask, each pixel is 0/1 representing background/foreground.
    '''
    def calculate_ari(self, gt_mask, pred_mask, gt_fg_mask):

        gt_mask = gt_mask.cpu().numpy()
        pred_mask = pred_mask.cpu().numpy()
        gt_fg_mask = gt_fg_mask.cpu().numpy()
        gt_sequence = gt_mask[np.where(gt_fg_mask>0)]
        pred_sequence = pred_mask[np.where(gt_fg_mask>0)]
        ari = adjusted_rand_score(pred_sequence, gt_sequence)

        return ari
    

    '''
    This function is to calculate AP
    INPUT:
    - gt_match_score: list of length [gt_ins_count], the IOU of each GT component with its matched prediction component
    - pred_match_score: list of length [pred_ins_count], the IOU of each predicted component with its matched GT component
    - pred_conf_score: list of length [pred_ins_count], confidence scores for each predicted component.
    '''
    def calculate_ap(self, gt_match_score, pred_match_score, pred_conf_score):
        gt_match_score = torch.from_numpy(np.array(gt_match_score))
        pred_conf_score = torch.from_numpy(np.array(pred_conf_score))
        gt_match_score_sorted = torch.sort(gt_match_score, descending=True)[0]
        ## NOTE: sort pred_match_score with pred_conf_score
        pred_match_score_sorted_list = []
        for i in range(0, len(pred_match_score)):
            index = torch.argmax(pred_conf_score)
            pred_conf_score[index] = -1
            pred_match_score_sorted_list.append(pred_match_score[index])
        pred_match_score_sorted = torch.from_numpy(np.array(pred_match_score_sorted_list))
        
        pred_match = pred_match_score_sorted >= 0.5
        precisions = torch.cumsum(pred_match, dim=0) / (torch.arange(len(pred_match)) + 1)
        gt_match = gt_match_score_sorted >= 0.5
        recalls = torch.cumsum(pred_match, dim=0).type(torch.float32) / len(gt_match)
        assert len(recalls) == len(precisions)
        return self.integral_method(precisions, recalls).item()

    '''
    This function calculates AP with SORTED precision list and recall list
    This method same as coco
    '''
    def integral_method(self, prec, rec):
        mrec = torch.cat((torch.Tensor([0.]), rec, torch.Tensor([1.])))
        mprec = torch.cat((torch.Tensor([0.]), prec, torch.Tensor([0.])))
        for i in range(mprec.shape[0] - 1, 0, -1):
            mprec[i - 1] = torch.maximum(mprec[i - 1], mprec[i])
        
        index = torch.where(mrec[1:] != mrec[:-1])[0]
        ap = torch.sum((mrec[index + 1] - mrec[index]) * mprec[index + 1])
        return ap


    
    '''
    This function tries to match one of the predicted component with GT background
    return a component that:
    1. in the valid prediction area
    2. has a iou with gt bg larger than 0.5
    '''
    def get_matched_bg(self,
            gt_mask,
            pred_mask,
            gt_fg_mask,
            valid_pred_mask,
            ):
        pred_mask = convert_to_tensor(pred_mask)
        gt_mask = convert_to_tensor(gt_mask)
        valid_pred_mask = convert_to_tensor(valid_pred_mask) if valid_pred_mask is not None else valid_pred_mask
        gt_fg_mask = convert_to_tensor(gt_fg_mask)
        match_bg_mask = torch.zeros([valid_pred_mask.shape[0], valid_pred_mask.shape[1]])

        ## try to match GT background with predicted component, 
        ## successful match when iou > 0.5
        ## if there is a match, remove that predicted component before AP clculation 
        pred_fg_mask = valid_pred_mask.clone()
        gt_bg_mask = 1-gt_fg_mask
        for pred_idx in torch.unique(torch.masked_select(pred_mask, valid_pred_mask.bool())):
            pred_idx = pred_idx.item()
            pred_ins = (pred_mask==pred_idx) * valid_pred_mask
            iou = torch.sum(pred_ins*gt_bg_mask) / (torch.sum(pred_ins) + torch.sum(gt_bg_mask) + 1e-6 - torch.sum(pred_ins*gt_bg_mask))
            if iou >= 0.5:
                match_bg_mask = valid_pred_mask * pred_ins
        
        return match_bg_mask