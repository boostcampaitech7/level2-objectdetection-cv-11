# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
import math
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import mmcv
import numpy as np
from mmcv.image import imresize
from mmcv.image.geometric import _scale_size
from mmcv.transforms import BaseTransform
from mmcv.transforms import Pad as MMCV_Pad
from mmcv.transforms import RandomFlip as MMCV_RandomFlip
from mmcv.transforms import Resize as MMCV_Resize
from mmcv.transforms.utils import avoid_cache_randomness, cache_randomness
from mmengine.dataset import BaseDataset
from mmengine.utils import is_str
from numpy import random

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import HorizontalBoxes, autocast_box_type
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from mmdet.utils import log_img_scale


@TRANSFORMS.register_module()
class custom_CopyPaste(BaseTransform):
    """Simple Copy-Paste is a Strong Data Augmentation Method for Instance
    Segmentation The simple copy-paste transform steps are as follows:

    1. The destination image is already resized with aspect ratio kept,
       cropped and padded.
    2. Randomly select a source image, which is also already resized
       with aspect ratio kept, cropped and padded in a similar way
       as the destination image.
    3. Randomly select some objects from the source image.
    4. Paste these source objects to the destination image directly,
       due to the source and destination image have the same size.
    5. Update object masks of the destination image, for some origin objects
       may be occluded.
    6. Generate bboxes from the updated destination masks and
       filter some objects which are totally occluded, and adjust bboxes
       which are partly occluded.
    7. Append selected source bboxes, masks, and labels.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)
    - gt_masks (BitmapMasks) (optional)

    Modified Keys:

    - img
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)
    - gt_masks (optional)

    Args:
        max_num_pasted (int): The maximum number of pasted objects.
            Defaults to 100.
        bbox_occluded_thr (int): The threshold of occluded bbox.
            Defaults to 10.
        mask_occluded_thr (int): The threshold of occluded mask.
            Defaults to 300.
        selected (bool): Whether select objects or not. If select is False,
            all objects of the source image will be pasted to the
            destination image.
            Defaults to True.
        paste_by_box (bool): Whether use boxes as masks when masks are not
            available.
            Defaults to False.
    """

    def __init__(
        self,
        max_num_pasted: int = 100,
        bbox_occluded_thr: int = 10,
        mask_occluded_thr: int = 300,
        selected: bool = True,
        paste_by_box: bool = True,
    ) -> None:
        self.max_num_pasted = max_num_pasted
        self.bbox_occluded_thr = bbox_occluded_thr
        self.mask_occluded_thr = mask_occluded_thr
        self.selected = selected
        self.paste_by_box = paste_by_box

    @cache_randomness
    def get_indexes(self, dataset: BaseDataset) -> int:
        """Call function to collect indexes.s.

        Args:
            dataset (:obj:MultiImageMixDataset): The dataset.
        Returns:
            list: Indexes.
        """
        return random.randint(0, len(dataset))

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Transform function to make a copy-paste of image.

        Args:
            results (dict): Result dict.
        Returns:
            dict: Result dict with copy-paste transformed.
        """

        assert 'mix_results' in results
        num_images = len(results['mix_results'])
        assert num_images == 1, \
            f'CopyPaste only supports processing 2 images, got {num_images}'
        if self.selected:
            selected_results = self._select_object(results['mix_results'][0])
        else:
            selected_results = results['mix_results'][0]
        return self._copy_paste(results, selected_results)

    @cache_randomness
    def _get_selected_inds(self, num_bboxes: int) -> np.ndarray:
        max_num_pasted = min(num_bboxes + 1, self.max_num_pasted)
        num_pasted = np.random.randint(0, max_num_pasted)
        return np.random.choice(num_bboxes, size=num_pasted, replace=False)

    def get_gt_masks(self, results: dict) -> BitmapMasks:
        """Get gt_masks originally or generated based on bboxes.

        If gt_masks is not contained in results,
        it will be generated based on gt_bboxes.
        Args:
            results (dict): Result dict.
        Returns:
            BitmapMasks: gt_masks, originally or generated based on bboxes.
        """
        if results.get('gt_masks', None) is not None:
            if self.paste_by_box:
                warnings.warn('gt_masks is already contained in results, '
                              'so paste_by_box is disabled.')
            return results['gt_masks']
        else:
            if not self.paste_by_box:
                raise RuntimeError('results does not contain masks.')
            return results['gt_bboxes'].create_masks(results['img'].shape[:2])

    def _select_object(self, results: dict) -> dict:
        """Select some objects from the source results."""
        bboxes = results['gt_bboxes']
        labels = results['gt_bboxes_labels']
        masks = self.get_gt_masks(results)
        ignore_flags = results['gt_ignore_flags']

        selected_inds = self._get_selected_inds(bboxes.shape[0])

        selected_bboxes = bboxes[selected_inds]
        selected_labels = labels[selected_inds]
        selected_masks = masks[selected_inds]
        selected_ignore_flags = ignore_flags[selected_inds]

        results['gt_bboxes'] = selected_bboxes
        results['gt_bboxes_labels'] = selected_labels
        results['gt_masks'] = selected_masks
        results['gt_ignore_flags'] = selected_ignore_flags
        return results

    def _copy_paste(self, dst_results: dict, src_results: dict) -> dict:
        """CopyPaste transform function.

        Args:
            dst_results (dict): Result dict of the destination image.
            src_results (dict): Result dict of the source image.
        Returns:
            dict: Updated result dict.
        """
        dst_img = dst_results['img']
        dst_bboxes = dst_results['gt_bboxes']
        dst_labels = dst_results['gt_bboxes_labels']
        dst_masks = self.get_gt_masks(dst_results)
        dst_ignore_flags = dst_results['gt_ignore_flags']

        src_img = src_results['img']
        src_bboxes = src_results['gt_bboxes']
        src_labels = src_results['gt_bboxes_labels']
        src_masks = src_results['gt_masks']
        src_ignore_flags = src_results['gt_ignore_flags']

        if len(src_bboxes) == 0:
            return dst_results

        # update masks and generate bboxes from updated masks
        composed_mask = np.where(np.any(src_masks.masks, axis=0), 1, 0)
        self.crop(composed_mask)
        
        updated_dst_masks = self._get_updated_masks(dst_masks, composed_mask)
        updated_dst_bboxes = updated_dst_masks.get_bboxes(type(dst_bboxes))
        assert len(updated_dst_bboxes) == len(updated_dst_masks)

        # filter totally occluded objects
        l1_distance = (updated_dst_bboxes.tensor - dst_bboxes.tensor).abs()
        bboxes_inds = (l1_distance <= self.bbox_occluded_thr).all(
            dim=-1).numpy()
        masks_inds = updated_dst_masks.masks.sum(
            axis=(1, 2)) > self.mask_occluded_thr
        valid_inds = bboxes_inds | masks_inds

        # Paste source objects to destination image directly
        img = dst_img * (1 - composed_mask[..., np.newaxis]
                         ) + src_img * composed_mask[..., np.newaxis] #dst_img는 mask없는부분이 나오고, src_img는 mask 있는 부분이 나온다..
        bboxes = src_bboxes.cat([updated_dst_bboxes[valid_inds], src_bboxes])
        labels = np.concatenate([dst_labels[valid_inds], src_labels])
        masks = np.concatenate(
            [updated_dst_masks.masks[valid_inds], src_masks.masks])
        ignore_flags = np.concatenate(
            [dst_ignore_flags[valid_inds], src_ignore_flags])

        dst_results['img'] = img
        dst_results['gt_bboxes'] = bboxes
        dst_results['gt_bboxes_labels'] = labels
        dst_results['gt_masks'] = BitmapMasks(masks, masks.shape[1],
                                              masks.shape[2])
        dst_results['gt_ignore_flags'] = ignore_flags

        return dst_results

    def _get_updated_masks(self, masks: BitmapMasks,
                           composed_mask: np.ndarray) -> BitmapMasks:
        """Update masks with composed mask."""
        assert masks.masks.shape[-2:] == composed_mask.shape[-2:], \
            'Cannot compare two arrays of different size'
        masks.masks = np.where(composed_mask, 0, masks.masks)
        return masks

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(max_num_pasted={self.max_num_pasted}, '
        repr_str += f'bbox_occluded_thr={self.bbox_occluded_thr}, '
        repr_str += f'mask_occluded_thr={self.mask_occluded_thr}, '
        repr_str += f'selected={self.selected}), '
        repr_str += f'paste_by_box={self.paste_by_box})'
        return repr_str

    def crop(self,img,mask):
        m, n = mask.shape
        mask0, mask1 = mask.any(axis = 0), mask.any(axis = 1)
        col_start, col_end = mask0.argmax(), n-mask0[::-1].argmax()
        row_start, row_end = mask1.argmax(), m-mask1[::-1].argmax()
        return img[max(0,row_start):row_end, max(0, col_start):col_end]

@TRANSFORMS.register_module()
class custom_mosaic_copy_paste(BaseTransform):
    """Mosaic augmentation.

    Given 4 images, mosaic transform combines them into
    one output image. The output image is composed of the parts from each sub-
    image.

    .. code:: text

                        mosaic transform
                           center_x
                +------------------------------+
                |       pad        |  pad      |
                |      +-----------+           |
                |      |           |           |
                |      |  image1   |--------+  |
                |      |           |        |  |
                |      |           | image2 |  |
     center_y   |----+-------------+-----------|
                |    |   cropped   |           |
                |pad |   image3    |  image4   |
                |    |             |           |
                +----|-------------+-----------+
                     |             |
                     +-------------+

     The mosaic transform steps are as follows:

         1. Choose the mosaic center as the intersections of 4 images
         2. Get the left top image according to the index, and randomly
            sample another 3 images from the custom dataset.
         3. Sub image will be cropped if image is larger than mosaic patch

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)
    - mix_results (List[dict])

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)

    Args:
        img_scale (Sequence[int]): Image size before mosaic pipeline of single
            image. The shape order should be (width, height).
            Defaults to (640, 640).
        center_ratio_range (Sequence[float]): Center ratio range of mosaic
            output. Defaults to (0.5, 1.5).
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        pad_val (int): Pad value. Defaults to 114.
        prob (float): Probability of applying this transformation.
            Defaults to 1.0.
    """

    def __init__(self,
                 img_scale: Tuple[int, int] = (640, 640),
                 center_ratio_range: Tuple[float, float] = (0.5, 1.5),
                 bbox_clip_border: bool = True,
                 pad_val: float = 114.0,
                 prob: float = 1.0,
                 paste_by_box = True,
                 max_num_pasted: int = 100,) -> None:
        assert isinstance(img_scale, tuple)
        assert 0 <= prob <= 1.0, 'The probability should be in range [0,1]. ' \
                                 f'got {prob}.'

        log_img_scale(img_scale, skip_square=True, shape_order='wh')
        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.bbox_clip_border = bbox_clip_border
        self.pad_val = pad_val
        self.prob = prob
        self.paste_by_box = paste_by_box
        self.max_num_pasted = max_num_pasted

    @cache_randomness
    def get_indexes(self, dataset: BaseDataset) -> int:
        """Call function to collect indexes.

        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.

        Returns:
            list: indexes.
        """

        indexes = [random.randint(0, len(dataset)) for _ in range(4)]
        return indexes

    @cache_randomness
    def _get_selected_inds(self, num_bboxes: int) -> np.ndarray:
        max_num_pasted = min(num_bboxes + 1, self.max_num_pasted)
        num_pasted = np.random.randint(0, max_num_pasted)
        return np.random.choice(num_bboxes, size=num_pasted, replace=False)

    def get_gt_masks(self, results: dict) -> BitmapMasks:
        """Get gt_masks originally or generated based on bboxes.

        If gt_masks is not contained in results,
        it will be generated based on gt_bboxes.
        Args:
            results (dict): Result dict.
        Returns:
            BitmapMasks: gt_masks, originally or generated based on bboxes.
        """
        if results.get('gt_masks', None) is not None:
            if self.paste_by_box:
                warnings.warn('gt_masks is already contained in results, '
                              'so paste_by_box is disabled.')
            return results['gt_masks']
        else:
            if not self.paste_by_box:
                raise RuntimeError('results does not contain masks.')
            return results['gt_bboxes'].create_masks(results['img'].shape[:2])
    
    def _copy_paste(self, src_results: dict) -> dict:

        src_img = src_results['img']
        src_bboxes = src_results['gt_bboxes']
        src_labels = src_results['gt_bboxes_labels']
        src_masks = self.get_gt_masks(src_results)
        src_ignore_flags = src_results['gt_ignore_flags']

        # update masks and generate bboxes from updated masks
        selected_inds = self._get_selected_inds(src_bboxes.shape[0])

        selected_bboxes = src_bboxes[selected_inds]
        selected_labels = src_labels[selected_inds]
        selected_masks = src_masks[selected_inds]
        selected_ignore_flags = src_ignore_flags[selected_inds]

        # Paste source objects to destination image directly
        #img = src_img * composed_mask[..., np.newaxis]
        
        src_results['gt_bboxes'] = selected_bboxes
        src_results['gt_bboxes_labels'] = selected_labels
        src_results['gt_masks'] = selected_masks
        src_results['gt_ignore_flags'] = selected_ignore_flags
        return src_results

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Mosaic transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        if random.uniform(0, 1) > self.prob:
            return results

        assert 'mix_results' in results
        mosaic_bboxes = []
        mosaic_bboxes_labels = []
        mosaic_ignore_flags = []
        if len(results['img'].shape) == 3:
            mosaic_img = np.full( #밑바탕 그리기
                (int(self.img_scale[1] * 2), int(self.img_scale[0] * 2), 3),
                self.pad_val,
                dtype=results['img'].dtype)
        else:
            mosaic_img = np.full(
                (int(self.img_scale[1] * 2), int(self.img_scale[0] * 2)),
                self.pad_val,
                dtype=results['img'].dtype)

        # mosaic center x, y
        center_x = int( #중심을 기점으로
            random.uniform(*self.center_ratio_range) * self.img_scale[0])
        center_y = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_position = (center_x, center_y)

        loc_strs = ('center','top_left', 'top_right', 'bottom_left', 'bottom_right') #4부분으로 나누고
        for i, loc in enumerate(loc_strs):
            if loc == 'center':
                bg_results_patch = copy.deepcopy(results) #idx 이미지
                background_img = bg_results_patch['img']
                h_i, w_i = background_img.shape[:2]
                # keep_ratio resize
                scale_ratio_i = min(self.img_scale[1] / h_i,
                                    self.img_scale[0] / w_i)
                background_img = mmcv.imresize(
                    background_img, (int(w_i * scale_ratio_i*2), int(h_i * scale_ratio_i*2)))
                composed_mask = np.zeros_like(mosaic_img[:,:,0])
            
                gt_bboxes_i = bg_results_patch['gt_bboxes'] 
                gt_bboxes_labels_i = bg_results_patch['gt_bboxes_labels']
                gt_ignore_flags_i = bg_results_patch['gt_ignore_flags']


                gt_bboxes_i.rescale_([2*scale_ratio_i, 2*scale_ratio_i]) 
                mosaic_bboxes.append(gt_bboxes_i)
                mosaic_bboxes_labels.append(gt_bboxes_labels_i)
                mosaic_ignore_flags.append(gt_ignore_flags_i)
            else:
                results['mix_results'][i - 1]= self._copy_paste(results['mix_results'][i - 1] )
                results_patch = copy.deepcopy(results['mix_results'][i - 1]) #나머지 랜덤 3개
                mask_i = np.where(np.any(results['mix_results'][i - 1]['gt_masks'].masks,axis=0),1,0)

                img_i = results_patch['img']
                h_i, w_i = img_i.shape[:2] #이미지 h,w
                # keep_ratio resize
                scale_ratio_i = min(self.img_scale[1] / h_i,
                                    self.img_scale[0] / w_i) # resize하기 위해 짧은 변의 길이 구함
                img_i = mmcv.imresize(
                    img_i.astype('float'), (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i))) #짧은 변의 길이에 맞춰서 resize

                mask_i = mmcv.imresize(
                    mask_i.astype('float'), (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i))) #짧은 변의 길이에 맞춰서 resize

                # compute the combine parameters
                paste_coord, crop_coord = self._mosaic_combine(  
                    loc, center_position, img_i.shape[:2][::-1]) #loc, center, w, h
                x1_p, y1_p, x2_p, y2_p = paste_coord
                x1_c, y1_c, x2_c, y2_c = crop_coord
                # crop and paste image
                composed_mask[y1_p:y2_p, x1_p:x2_p] = mask_i[y1_c:y2_c, x1_c:x2_c]
                mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c] #모자이크에 맞춰서 들어감

                                # adjust coordinate
                gt_bboxes_i = results_patch['gt_bboxes'] 
                gt_bboxes_labels_i = results_patch['gt_bboxes_labels']
                gt_ignore_flags_i = results_patch['gt_ignore_flags']

                padw = x1_p - x1_c
                padh = y1_p - y1_c

                gt_bboxes_i.rescale_([scale_ratio_i, scale_ratio_i]) 
                gt_bboxes_i.translate_([padw, padh])
                
                mosaic_bboxes.append(gt_bboxes_i)
                mosaic_bboxes_labels.append(gt_bboxes_labels_i)
                mosaic_ignore_flags.append(gt_ignore_flags_i)

            mosaic_img = background_img * (1 - composed_mask[..., np.newaxis]
                    ) + mosaic_img * composed_mask[..., np.newaxis]


        mosaic_bboxes = mosaic_bboxes[0].cat(mosaic_bboxes, 0)
        mosaic_bboxes_labels = np.concatenate(mosaic_bboxes_labels, 0)
        mosaic_ignore_flags = np.concatenate(mosaic_ignore_flags, 0)

        if self.bbox_clip_border:
            mosaic_bboxes.clip_([2 * self.img_scale[1], 2 * self.img_scale[0]])
        # remove outside bboxes
        inside_inds = mosaic_bboxes.is_inside(
            [2 * self.img_scale[1], 2 * self.img_scale[0]]).numpy()
        mosaic_bboxes = mosaic_bboxes[inside_inds]
        mosaic_bboxes_labels = mosaic_bboxes_labels[inside_inds]
        mosaic_ignore_flags = mosaic_ignore_flags[inside_inds]

        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape[:2]
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_bboxes_labels'] = mosaic_bboxes_labels
        results['gt_ignore_flags'] = mosaic_ignore_flags
        return results

    def _mosaic_combine(
            self, loc: str, center_position_xy: Sequence[float],
            img_shape_wh: Sequence[int]) -> Tuple[Tuple[int], Tuple[int]]:
        """Calculate global coordinate of mosaic image and local coordinate of
        cropped sub-image.

        Args:
            loc (str): Index for the sub-image, loc in ('top_left',
              'top_right', 'bottom_left', 'bottom_right').
            center_position_xy (Sequence[float]): Mixing center for 4 images,
                (x, y).
            img_shape_wh (Sequence[int]): Width and height of sub-image

        Returns:
            tuple[tuple[float]]: Corresponding coordinate of pasting and
                cropping
                - paste_coord (tuple): paste corner coordinate in mosaic image.
                - crop_coord (tuple): crop corner coordinate in mosaic image.
        """
        assert loc in ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        if loc == 'top_left':
            # index0 to top left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             center_position_xy[0], \
                             center_position_xy[1]
            crop_coord = img_shape_wh[0] - (x2 - x1), img_shape_wh[1] - (
                y2 - y1), img_shape_wh[0], img_shape_wh[1]

        elif loc == 'top_right':
            # index1 to top right part of image
            x1, y1, x2, y2 = center_position_xy[0], \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[0] * 2), \
                             center_position_xy[1]
            crop_coord = 0, img_shape_wh[1] - (y2 - y1), min(
                img_shape_wh[0], x2 - x1), img_shape_wh[1]

        elif loc == 'bottom_left':
            # index2 to bottom left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             center_position_xy[1], \
                             center_position_xy[0], \
                             min(self.img_scale[1] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = img_shape_wh[0] - (x2 - x1), 0, img_shape_wh[0], min(
                y2 - y1, img_shape_wh[1])

        else:
            # index3 to bottom right part of image
            x1, y1, x2, y2 = center_position_xy[0], \
                             center_position_xy[1], \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[0] * 2), \
                             min(self.img_scale[1] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = 0, 0, min(img_shape_wh[0],
                                   x2 - x1), min(y2 - y1, img_shape_wh[1])

        paste_coord = x1, y1, x2, y2
        return paste_coord, crop_coord

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'center_ratio_range={self.center_ratio_range}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'prob={self.prob}), '
        repr_str += f'(max_num_pasted={self.max_num_pasted}, '
        repr_str += f'paste_by_box={self.paste_by_box})'
        return repr_str