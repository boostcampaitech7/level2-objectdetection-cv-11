_base_ = 'configs/yolo/yolov3_d53_8xb8-ms-608-273e_coco.py'
# fp16 settings
optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic')

#class 변경
metainfo = {
    'classes': ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"),
    'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30)]
}

img_scale=(1024, 1024)

pre_transform = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True)
]

train_pipeline = [
    *pre_transform,
    dict(type='Resize', scale = img_scale),
    dict(
        type='CopyPaste',
        paste_by_box = True
    ),
    dict(
        type='mmdet.RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(type='PackDetInputs')
]

data_root='/data/ephemeral/home/dataset'

train_dataset = dict(
            _delete_=True,
    # use MultiImageMixDataset wrapper to support mosaic and mixup
    type='MultiImageMixDataset',
    dataset=dict(
        type='CocoDataset',
        metainfo = metainfo,
        data_root=data_root,
        ann_file='train.json',
        data_prefix=dict(img= data_root),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ]),
    pipeline=train_pipeline)

train_dataloader = dict(
    batch_size=4,
    dataset=train_dataset)

val_dataloader = None
val_cfg = None
val_evaluator = None