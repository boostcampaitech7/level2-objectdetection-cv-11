_base_ = [
    '../../configs/_base_/datasets/coco_detection.py',
    '../../configs/_base_/default_runtime.py',
]

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa

### Data Preprocessor ###
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=1
)

### Model ###
model = dict(
    type='ATSS',
    data_preprocessor=data_preprocessor,  # data_preprocessor 추가
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)
    ),
    neck=[
        dict(
            type='FPN',
            in_channels=[384, 768, 1536],
            out_channels=256,
            start_level=0,
            add_extra_convs='on_output',
            num_outs=5
        ),
        dict(
            type='DyHead',
            in_channels=256,
            out_channels=256,
            num_blocks=6,
            zero_init_offset=False
        )
    ],
    bbox_head=dict(
        type='ATSSHead',
        num_classes=10,
        in_channels=256,
        pred_kernel_size=1,  # DyHead 공식 구현을 따름
        stacked_convs=0,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128],
            center_offset=0.5  # DyHead 공식 구현을 따름
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        ),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0
        )
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100
    )
)

# custom hooks
custom_hooks = [dict(type='SubmissionHook')]

### Optimizer ###
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))

### Learning Policy ###
max_epochs = 25

param_scheduler = [
    # dict(
    #     type='LinearLR',
    #     start_factor=0.001,
    #     by_epoch=False,
    #     begin=0,
    #     end=500
    # ),
    # dict(
    #     type='StepLR',
    #     step_size=8,
    #     gamma=0.1,
    #     by_epoch=True
    # )
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0,
        begin=0,
        T_max=max_epochs,
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True)    
]

### Runner Configuration ###
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

### Dataset ###
data_root = '/home/donghun0671/workplace/lv2/sr_dataset/'
data_root_test = '/home/donghun0671/workplace/lv2/dataset/'
metainfo = {
    'classes': (
        'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
        'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing',
    ),
    'palette': [
        (220, 20, 60), (119, 11, 32), (0, 0, 230), (106, 0, 228),
        (60, 20, 220), (0, 80, 100), (0, 0, 70), (50, 0, 192),
        (250, 170, 30), (255, 0, 0)
    ]
}

### Data Loaders ###
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='kfolds/integrated_train_kfold_0.json',
        data_prefix=dict(img=''),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=dict()),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='RandomResize',
                scale=(1024, 1024),
                ratio_range=(0.1, 2.0),
                keep_ratio=True
            ),
            dict(
                type='RandomCrop',
                crop_type='absolute_range',
                crop_size=(1024, 1024),
                recompute_bbox=True,
                allow_negative_crop=True
            ),
            # dict(type='RandAugment', aug_space=color_space, aug_num=1),
            dict(type='PackDetInputs')  # PackDetInputs 유지
        ]
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='kfolds/integrated_val_kfold_0.json',
        data_prefix=dict(img=''),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=dict()),
            dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',  # PackDetInputs 유지
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
            )
        ]
    )
)

test_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root_test,
        metainfo=metainfo,
        ann_file='test.json',
        data_prefix=dict(img=''),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=dict()),
            dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',  # PackDetInputs 유지
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
            )
        ]
    )
)

### Evaluators ###
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'kfolds/integrated_val_kfold_0.json',
    metric='bbox',
    format_only=False,
    classwise=True,
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root_test + 'test.json',
    metric='bbox',
    format_only=False,
    classwise=True,
)

### Visualizer ###
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            type='WandbVisBackend',
            init_kwargs=dict(
                entity='hanseungsoo63-naver',
                project='atss',
                name='atss_swin-l_dyhead_aug_3_ep25_crop_aug'
            )
        )
    ],
    name='visualizer'
)

### Auto Scale LR ###
auto_scale_lr = dict(base_batch_size=2)

### Load Checkpoint ###
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/dyhead/atss_swin-l-p4-w12_fpn_dyhead_mstrain_2x_coco/atss_swin-l-p4-w12_fpn_dyhead_mstrain_2x_coco_20220509_100315-bc5b6516.pth'