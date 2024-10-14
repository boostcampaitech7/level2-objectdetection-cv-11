_base_ = 'configs/yolo/yolov3_d53_8xb8-ms-608-273e_coco.py'
# fp16 settings
#optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic')

#class 변경
metainfo = {
    'classes': ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"),
    'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30)]
}

img_scale=(1024, 1024)

data_root='/data/ephemeral/home/dataset/'

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

#train data
train_dataloader = None
train_cfg = None
optim_wrapper = None
param_scheduler = None

val_dataloader = None
val_cfg = None
val_evaluator = None

test_dataloader = dict(
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        metainfo = metainfo,
        type=_base_.dataset_type,
        data_root=data_root,
        ann_file='evaluation_nonannot.json',
        data_prefix=dict(img=data_root),
        test_mode=True))

test_cfg = dict(type='TestLoop')

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'evaluation_nonannot.json',
    metric='bbox')