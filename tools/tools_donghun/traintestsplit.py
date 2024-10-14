import json
import os
from collections import Counter
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# 데이터 annotation 파일 저장 경로
save_annotation_path = '/home/donghun0671/workplace/lv2/dataset'
# annotation 파일 경로
annotation = '/home/donghun0671/workplace/lv2/dataset/train.json'

# 디렉토리가 존재하지 않으면 생성
os.makedirs(save_annotation_path, exist_ok=True)

with open(annotation, 'r', encoding='utf-8') as f:
    data = json.load(f)
    info = data['info']
    licences = data['licenses']
    images = data['images']
    categories = data['categories']
    anns = data['annotations']

# COCO 형식으로 파일을 저장하는 함수
def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'w', encoding='utf-8') as coco_file:
        json.dump({
            'info': info,
            'licenses': licenses,
            'images': images,
            'annotations': annotations,
            'categories': categories
        }, coco_file, indent=2, ensure_ascii=False)

# annotations를 필터링하는 함수
def filter_annotations(annotations, image_ids):
    return [ann for ann in annotations if ann['image_id'] in image_ids]

# images를 필터링하는 함수
def filter_images(images, image_ids):
    return [img for img in images if img['id'] in image_ids]

# 이미지 ID별 카테고리 리스트 생성
image_to_categories = {}
for ann in anns:
    image_id = ann['image_id']
    category_id = ann['category_id']
    if image_id not in image_to_categories:
        image_to_categories[image_id] = []
    image_to_categories[image_id].append(category_id)

# 고유 이미지 ID 리스트
unique_image_ids = [img['id'] for img in images]

# 분할을 위한 그룹 정보
groups = np.array(unique_image_ids)

# 각 그룹(이미지)에 대한 레이블 (가장 많이 등장하는 카테고리)
y = np.array([Counter(image_to_categories[img_id]).most_common(1)[0][0] for img_id in unique_image_ids])

# GroupShuffleSplit을 사용하여 80:20 분할
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=137)
train_idx, val_idx = next(splitter.split(unique_image_ids, y, groups))

train_image_ids = [unique_image_ids[i] for i in train_idx]
val_image_ids = [unique_image_ids[i] for i in val_idx]

# 필터링된 이미지 및 annotation 가져오기
train_images = filter_images(images, train_image_ids)
val_images = filter_images(images, val_image_ids)

train_annotations = filter_annotations(anns, train_image_ids)
val_annotations = filter_annotations(anns, val_image_ids)

# COCO 형식으로 저장
train_file_name = 'train_split.json'
val_file_name = 'val_split.json'

save_coco(os.path.join(save_annotation_path, train_file_name), info, licences, train_images, train_annotations, categories)
print(f'Training annotation saved as {train_file_name}')

save_coco(os.path.join(save_annotation_path, val_file_name), info, licences, val_images, val_annotations, categories)
print(f'Validation annotation saved as {val_file_name}')


# import json
# import os
# import funcy
# import numpy as np
# from sklearn.model_selection import GroupShuffleSplit

# # 데이터 annotation 파일 저장 경로
# save_annotation_path = '/home/donghun0671/workplace/lv2/dataset'
# # annotation 파일 경로
# annotation = '/home/donghun0671/workplace/lv2/dataset/train.json'

# # 디렉토리가 존재하지 않으면 생성
# os.makedirs(save_annotation_path, exist_ok=True)

# with open(annotation, 'r', encoding='utf-8') as f:
#     data = json.load(f)
#     info = data['info']
#     licences = data['licenses']
#     images = data['images']
#     categories = data['categories']
#     anns = data['annotations']

# # COCO 형식으로 파일을 저장하는 함수
# def save_coco(file, info, licenses, images, annotations, categories):
#     with open(file, 'w', encoding='utf-8') as coco_file:
#         json.dump({
#             'info': info,
#             'licenses': licenses,
#             'images': images,
#             'annotations': annotations,
#             'categories': categories
#         }, coco_file, indent=2, ensure_ascii=False)

# # annotations를 필터링하는 함수
# def filter_annotations(annotations, image_ids):
#     return [ann for ann in annotations if ann['image_id'] in image_ids]

# # images를 필터링하는 함수
# def filter_images(images, image_ids):
#     return [img for img in images if img['id'] in image_ids]

# # 이미지 ID별 카테고리 리스트 생성
# image_to_categories = {}
# for ann in anns:
#     image_id = ann['image_id']
#     category_id = ann['category_id']
#     if image_id not in image_to_categories:
#         image_to_categories[image_id] = []
#     image_to_categories[image_id].append(category_id)

# # 고유 이미지 ID 리스트
# unique_image_ids = [img['id'] for img in images]

# # 분할을 위한 그룹 정보
# groups = np.array(unique_image_ids)

# # 각 그룹(이미지)에 대한 레이블 (가장 많이 등장하는 카테고리)
# y = np.array([funcy.mode(image_to_categories[img_id])[0] for img_id in unique_image_ids])

# # GroupShuffleSplit을 사용하여 80:20 분할
# splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=137)
# train_idx, val_idx = next(splitter.split(unique_image_ids, y, groups))

# train_image_ids = [unique_image_ids[i] for i in train_idx]
# val_image_ids = [unique_image_ids[i] for i in val_idx]

# # 필터링된 이미지 및 annotation 가져오기
# train_images = filter_images(images, train_image_ids)
# val_images = filter_images(images, val_image_ids)

# train_annotations = filter_annotations(anns, train_image_ids)
# val_annotations = filter_annotations(anns, val_image_ids)

# # COCO 형식으로 저장
# train_file_name = 'train_split.json'
# val_file_name = 'val_split.json'

# save_coco(os.path.join(save_annotation_path, train_file_name), info, licences, train_images, train_annotations, categories)
# print(f'Training annotation saved as {train_file_name}')

# save_coco(os.path.join(save_annotation_path, val_file_name), info, licences, val_images, val_annotations, categories)
# print(f'Validation annotation saved as {val_file_name}')
