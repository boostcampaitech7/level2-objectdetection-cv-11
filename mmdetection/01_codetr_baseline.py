# 기본 설정 파일 로드
_base_ = './projects/CO-DETR/configs/codino/co_dino_5scale_r50_lsj_8xb2_1x_coco.py'

num_dec_layer = 6  # DETR(Detection Transformer) 사용하는 디코더 레이어 수
loss_lambda = 2.0  # 손실 함수(loss function)**에서 사용하는 가중치 값
num_classes = 10  # numclass

# 모델 정의
model = dict(
    data_preprocessor=dict(
        # 객체 탐지 사용을 위한 설정
        pad_mask=False,  # mask 사용 false
        batch_augments=None  # 배치 단위 증강 none
    )
)

# class 변경
# palette 가 있는게 좋은가??
metainfo = {
    'classes': ("General trash", "Paper", "Paper pack", "Metal", "Glass",
                "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"),
    'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
                (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30)]
}

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),  # 파일로부터 이미지를 로드
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),  # 바운딩 박스 정보를 포함한 어노테이션 로드, mask 는 false
    dict(type='RandomFlip', prob=0.5),  # 50% 확률로 이미지를 좌우 반전

    # 첫 번째 그룹이 선택되면, 이미지가 랜덤 크기로 리사이즈
    # 두 번째 그룹이 선택되면, 먼저 큰 사이즈로 리사이즈한 후, 이미지의 일부를 랜덤하게 잘라내고, 다시 특정 크기 중 하나로 리사이즈
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

    # 이미지와 어노테이션 정보가 묶여서 모델이 요구하는 형식으로 전달
    dict(type='PackDetInputs')  # 모델에 입력하기 위해 데이터를 포장
]

data_root = '../dataset'  # dataset root

# train data
train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        metainfo=metainfo,  # 클래스 정보와 팔레트 설정
        type=_base_.dataset_type,  # 사용할 데이터셋의 유형, 기본 설정에서 정의된 dataset_type 사용
        data_root=data_root,  # dataset root
        ann_file='train.json',  # 학습에 사용할 어노테이션
        data_prefix=dict(img=data_root),  # 이미지 파일이 위치한 경로
        filter_cfg=dict(filter_empty_gt=False, min_size=32),  # 필터 설정
        pipeline=train_pipeline,  # augmentation
        backend_args=_base_.backend_args),  # 데이터 로드 방식 정의
    batch_size=2)  # 배치사이즈 설정

# 추론 할때 사용
# test_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
#     dict(type='Resize', scale=(1333, 800), keep_ratio=True),
#     dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
#     dict(
#         type='PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                    'scale_factor'))
# ]

# validation, test 없음
val_dataloader = None
val_cfg = None
val_evaluator = None
test_dataloader = None
test_cfg = None
test_evaluator = None


# auto_scale_lr = dict(base_batch_size=16)
work_dir = './work_dirs/codetr_r50_trash'  # 저장할 workdir


# 궁금증
# 왜 vaildation 은 사용 안했는지
# pallate 를 사용하면 더 좋은가??


#
