_base_ = './projects/CO-DETR/configs/codino/co_dino_5scale_r50_lsj_8xb2_1x_coco.py'

#pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa

num_dec_layer = 6
loss_lambda = 2.0
num_classes = 10
# Co-Detr_SwinL_12_DETR 정의
model = dict(
    use_lsj=False, data_preprocessor=dict(pad_mask=False, batch_augments=None))


# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.

#class 변경
metainfo = {
    'classes': ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"),
    'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30)]
}

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs')
]

data_root='/data/ephemeral/home/dataset'

#train data
train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        _delete_=True,
        metainfo = metainfo,
        type=_base_.dataset_type,
        data_root=data_root,
        ann_file='train.json',
        data_prefix=dict(img= data_root),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline,
        backend_args=_base_.backend_args))

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1280), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

#validation, test 없음
val_dataloader = None
val_cfg = None
val_evaluator = None
test_dataloader = None
test_cfg = None
test_evaluator = None

#Wandb 연결
vis_backends = [
    dict(_scope_='mmdet', type='LocalVisBackend'),
    dict(type = 'TensorboardVisBackend'),
    dict(type = 'WandbVisBackend',
    init_kwargs = dict(
        project = 'hanseungsoo63-naver',
        group = 'co_detr_5scale_coco',
    ))
]

visualizer = dict(
    type = 'DetLocalVisualizer',
    vis_backends = vis_backends,
    name = 'visualizer'
)

auto_scale_lr = dict(base_batch_size=16)
work_dir = './work_dirs/codetr_swinl_r50_trash'

max_epochs=16

train_cfg = dict(max_epochs=max_epochs)