# 기본 설정 파일 로드
_base_ = './configs/yolo/atss_r101_fpn_8xb8-amp-lsj-200e_coco.py'

num_classes = 10

# 모델 정의
model = dict(
    bbox_head=dict(
        num_classes=num_classes  # 10개 클래스에 맞게 설정
    ),
    data_preprocessor=dict(
        # 객체 탐지 사용을 위한 설정
        pad_mask=False,  # mask 사용 false
        batch_augments=None  # 배치 단위 증강 none
    )
)

# class 변경
metainfo = {
    'classes': ("General trash", "Paper", "Paper pack", "Metal", "Glass",
                "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"),
}

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),  # 파일로부터 이미지를 로드
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),  # 바운딩 박스 정보를 포함한 어노테이션 로드, mask 는 false
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='PackDetInputs')  # 모델에 입력하기 위해 데이터를 포장
]

val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),  # 파일로부터 이미지를 로드
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),  # 바운딩 박스 정보를 포함한 어노테이션 로드, mask 는 false
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
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
        ann_file='./split/train.json',  # 학습에 사용할 어노테이션
        data_prefix=dict(img=data_root),  # 이미지 파일이 위치한 경로
        filter_cfg=dict(filter_empty_gt=False, min_size=32),  # 필터 설정
        pipeline=train_pipeline,  # augmentation
        backend_args=_base_.backend_args),  # 데이터 로드 방식 정의
    batch_size=4  # 배치사이즈 설정
)

# training configuration 추가
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,
    val_interval=1  # 매 1 에폭마다 검증을 실행하도록 설정
)


# validation data 설정
val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,  # 클래스 정보
        type='CocoDataset',  # COCO 형식 데이터셋 사용
        data_root=data_root,  # 검증 데이터 루트
        ann_file='./split/val.json',  # 검증 데이터 어노테이션 파일
        data_prefix=dict(img=data_root),  # 이미지 경로 설정
        filter_cfg=dict(filter_empty_gt=False, min_size=32),  # 필터 설정
        pipeline=val_pipeline,  # validation augmentation
    ),
    batch_size=4,
    num_workers=2,  # 데이터 로딩에 사용할 프로세스의 개수
    persistent_workers=True,
    drop_last=False,  # 배치 크기가 작아도 포함하여 학습
    sampler=dict(type='DefaultSampler', shuffle=False),  # 데이터셋 로딩 순서 설정
)

# validation configuration 설정
val_cfg = dict()  # 빈 dict로 설정해도 문제가 없음

# validation evaluator 설정
val_evaluator = dict(
    type='CocoMetric',
    ann_file='../dataset/split/val.json',
    metric=['bbox'],  # 필요한 metric 설정
    format_only=False,
    iou_thrs=[0.50, 0.75],  # Iou 성능 평가 기준
)


# test 없음
test_dataloader = None
test_cfg = None
test_evaluator = None

work_dir = './work_dirs/atss'  # 저장할 workdir
