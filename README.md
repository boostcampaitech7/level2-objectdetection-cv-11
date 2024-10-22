## Copy Paste 활용
1. `custom`폴더의 `copy_paste_aug` 폴더를 `mmdetection/mmdet`에 넣어줍니다.
2. `mmdet/datasets/transforms/__init__.py`에 아래와 같은 두 코드를 넣어줍니다.
   ```
   from mmdet.copy_paste_aug.custom_loader import LoadAnnotationswithbox
   from mmdet.copy_paste_aug.custom_copypaste import custom_CopyPaste
   ```
3. `__all__` 리스트에 `'custom_CopyPaste','LoadAnnotationswithbox'` 추가

## Train
1. `train.py`의 train_pipeline 수정
   ```
   train_pipeline = [
    dict(
        type='custom_mosaic_copy_paste',paste_by_box=True),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        skip_img_without_anno=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(50, 50), keep_empty=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type = 'RandomCrop',crop_size=img_scale),
    dict(type='PackDetInputs')
    ]
   ```
2. `train.py`의 train_dataset 수정
   ```
   train_dataset = dict(
        _delete_=True,
    # use MultiImageMixDataset wrapper to support mosaic and mixup
    type='MultiImageMixDataset',
    dataset=dict(
        type='CocoDataset',
        metainfo = metainfo,
        data_root=data_root,
        ann_file='train_2_annot.json',
        data_prefix=dict(img= ''),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ]),
    pipeline=train_pipeline)
   ```

자세한 사항은 `ddq_train.py` 참고
