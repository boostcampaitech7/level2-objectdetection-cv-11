# %%
import json
import os
import random
import shutil
from pycocotools.coco import COCO

# 설정
data_dir = '../dataset'  # 데이터가 저장된 디렉토리
coco_file = os.path.join(data_dir, 'train.json')  # COCO 형식의 JSON 파일
output_dir = os.path.join(data_dir, 'split')  # 결과를 저장할 디렉토리
train_ratio = 0.8  # 훈련 데이터 비율

# COCO 데이터셋 로드
with open(coco_file, 'r') as f:
    coco_data = json.load(f)

# 이미지 목록과 갯수
images = coco_data['images']
num_images = len(images)

# 랜덤하게 섞기
random.seed(42)  # 재현 가능성을 위해 시드 설정
random.shuffle(images)

# 훈련 데이터와 검증 데이터 나누기
train_count = int(num_images * train_ratio)
train_images = images[:train_count]
val_images = images[train_count:]

# 훈련 및 검증 데이터를 저장할 디렉토리 생성
os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)

# %%

# 훈련 데이터 저장
train_data = {
    'info': coco_data['info'],
    'licenses': coco_data['licenses'],
    'categories': coco_data['categories'],  # categories 추가
    'images': train_images,
    'annotations': []
}
for img in train_images:
    img_id = img['id']
    train_data['annotations'].extend([ann for ann in coco_data['annotations'] if ann['image_id'] == img_id])

with open(os.path.join(output_dir,  'train.json'), 'w') as f:
    json.dump(train_data, f)

# %%
# 검증 데이터 저장
val_data = {
    'info': coco_data['info'],
    'licenses': coco_data['licenses'],
    'categories': coco_data['categories'],  # categories 추가
    'images': val_images,
    'annotations': []
}
for img in val_images:
    img_id = img['id']
    val_data['annotations'].extend([ann for ann in coco_data['annotations'] if ann['image_id'] == img_id])

with open(os.path.join(output_dir, 'val.json'), 'w') as f:
    json.dump(val_data, f)

# %%

# train 이미지 파일 복사
for img in train_images:
    img_filename = img['file_name']  # 이미지 파일 이름 가져오기
    src_path = os.path.join(data_dir, img_filename)  # 원본 경로
    dst_path = os.path.join(data_dir, 'split', 'train', img_filename)  # 복사할 경로 (train 폴더)

    # 원본 파일이 존재하는지 확인하고 복사
    if os.path.exists(src_path):
        # 대상 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(src_path, dst_path)  # 파일 복사
        # print(f"복사 완료: {src_path} -> {dst_path}")
    else:
        print(f"파일을 찾을 수 없습니다: {src_path}")

print("Train 이미지 복사가 완료되었습니다!")

# %%
# val 이미지 파일 복사
for img in val_images:
    img_filename = img['file_name']  # 이미지 파일 이름 가져오기
    src_path = os.path.join(data_dir, img_filename)  # 원본 경로
    dst_path = os.path.join(data_dir, 'split', 'val', os.path.basename(img_filename))  # 복사할 경로 (val 폴더)

    # 원본 파일이 존재하는지 확인하고 복사
    if os.path.exists(src_path):
        # 대상 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(src_path, dst_path)  # 파일 복사
        # print(f"복사 완료: {src_path} -> {dst_path}")
    else:
        print(f"파일을 찾을 수 없습니다: {src_path}")

print("Val 이미지 복사가 완료되었습니다!")

print("Train/Val 데이터셋이 성공적으로 분할되었습니다!")
