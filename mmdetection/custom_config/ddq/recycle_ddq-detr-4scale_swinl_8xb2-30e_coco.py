_base_ = [
    '../../configs/_base_/datasets/coco_detection_trash_aug_3.py', '../../configs/_base_/default_runtime.py'
]
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa: E501
model = dict(
    type='DDQDETR',
    num_queries=900,  # num_matching_queries
    # ratio of num_dense queries to num_queries
    dense_topk_ratio=1.5,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
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
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='ChannelMapper',
        in_channels=[384, 768, 1536],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    # encoder class name: DeformableDetrTransformerEncoder
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4,
                               dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0))),  # 0.1 for DeformDETR
    # decoder class name: DDQTransformerDecoder
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8,
                               dropout=0.0),  # 0.1 for DeformDETR
            cross_attn_cfg=dict(embed_dims=256, num_levels=4,
                                dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0)),  # 0.1 for DeformDETR
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,  # -0.5 for DeformDETR
        temperature=20),  # 10000 for DeformDETR
    bbox_head=dict(
        type='DDQDETRHead',
        num_classes=80,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
    dqs_cfg=dict(type='nms', iou_threshold=0.8),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300))


# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.05),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.05)}))

# learning policy
max_epochs = 30
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=False,
        begin=0,
        end=2000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[20, 26],
        gamma=0.1)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=2)


default_hooks = dict(visualization=dict(type="DetVisualizationHook",draw=True))

# custom hooks
custom_hooks = [dict(type='SubmissionHook')]

# data_root = '/data/ephemeral/home/dataset/'
data_root = '/home/donghun0671/workplace/lv2/dataset/'

metainfo = {
    'classes': ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing',),
    'palette': [
        (220, 20, 60), (119, 11, 32), (0, 0, 230), (106, 0, 228), (60, 20, 220),
        (0, 80, 100), (0, 0, 70), (50, 0, 192), (250, 170, 30), (255, 0, 0)
    ]
}

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train_split.json',
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
        ]))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val_split.json',
        data_prefix=dict(img=''),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=dict()),
            dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',  # PackDetInputs 유지
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
            )
        ]))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
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
        ]))

### Evaluators ###
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val_split.json',
    metric='bbox',
    format_only=False,
    classwise=True,
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test.json',
    metric='bbox',
    format_only=False,
    classwise=True,
)

visualizer = dict(
    type='DetLocalVisualizer',    
    vis_backends = [
        dict(type='LocalVisBackend'), 
        dict(type='WandbVisBackend',
             init_kwargs=dict(
                 entity='hanseungsoo63-naver',
                 project='ddq',
                 name='recycle_ddq-detr-4scale_r50_8xb2-12e_coco'))],
    name='visualizer'    
    )


# load_from = "https://download.openmmlab.com/mmdetection/v3.0/ddq/ddq-detr-4scale_r50_8xb2-12e_coco/ddq-detr-4scale_r50_8xb2-12e_coco_20230809_170711-42528127.pth"
load_from = "https://download.openmmlab.com/mmdetection/v3.0/ddq/ddq_detr_swinl_30e.pth"