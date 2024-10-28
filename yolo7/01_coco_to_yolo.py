import os
import json
import shutil


def coco_to_yolo(coco_annotation_file, output_dir, images_dir):
    with open(coco_annotation_file) as f:
        coco = json.load(f)

    # 클래스 ID 매핑
    categories = {cat['id']: cat['name'] for cat in coco['categories']}

    # 출력 디렉토리 생성 (존재하지 않으면 생성)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img in coco['images']:
        image_id = img['id']
        image_file = img['file_name']
        height, width = img['height'], img['width']

        annotations = [ann for ann in coco['annotations'] if ann['image_id'] == image_id]

        # 라벨 디렉토리 경로 설정
        label_file_path = os.path.join(output_dir, os.path.splitext(image_file)[0] + ".txt")
        label_dir = os.path.dirname(label_file_path)

        # 라벨 디렉토리 존재 여부 확인 및 생성
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        with open(label_file_path, 'w') as f:
            for ann in annotations:
                category_id = ann['category_id']
                bbox = ann['bbox']  # [x_min, y_min, width, height]
                x_min, y_min, bbox_width, bbox_height = bbox

                # YOLO 형식으로 변환
                x_center = (x_min + bbox_width / 2) / width
                y_center = (y_min + bbox_height / 2) / height
                norm_width = bbox_width / width
                norm_height = bbox_height / height

                f.write(f"{category_id} {x_center} {y_center} {norm_width} {norm_height}\n")

        # 이미지 파일 복사
        shutil.copy(os.path.join(images_dir, image_file), os.path.join(output_dir, image_file))


# COCO 데이터셋 변환 실행
coco_to_yolo('../dataset/train.json', './labels', '../dataset')
