# 기본 설정 파일 로드
_base_ = './configs/yolo/yolov3_d53_8xb8-320-273e_coco.py'

# 모델 정의
model = dict(
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
    batch_size=2  # 배치사이즈 설정
)

# 훈련 설정 추가
train_cfg = dict(
    type='EpochBasedTrainLoop',  # 에폭 기반 훈련 루프 사용
    max_epochs=50,  # 설정할 에폭 수
)

# validation, test 없음
val_dataloader = None
val_cfg = None
val_evaluator = None
test_dataloader = None
test_cfg = None
test_evaluator = None

work_dir = './work_dirs/yolo'  # 저장할 workdir
