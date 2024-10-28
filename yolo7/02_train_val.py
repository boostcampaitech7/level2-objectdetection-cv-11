import json
import os
import shutil
from sklearn.model_selection import train_test_split


def split_dataset(yolo_images_dir, yolo_labels_dir, output_train_dir, output_val_dir, test_size=0.2):
    image_files = [f for f in os.listdir(yolo_images_dir) if f.endswith('.jpg')]

    # 데이터셋을 8:2로 분리
    train_images, val_images = train_test_split(image_files, test_size=test_size, random_state=42)

    for img_set, output_dir in [(train_images, output_train_dir), (val_images, output_val_dir)]:
        image_output_dir = os.path.join(output_dir, 'images')
        label_output_dir = os.path.join(output_dir, 'labels')

        os.makedirs(image_output_dir, exist_ok=True)
        os.makedirs(label_output_dir, exist_ok=True)

        for image_file in img_set:
            # 이미지 복사
            shutil.copy(os.path.join(yolo_images_dir, image_file), os.path.join(image_output_dir, image_file))

            # 라벨 파일 복사
            label_file = os.path.splitext(image_file)[0] + '.txt'
            shutil.copy(os.path.join(yolo_labels_dir, label_file), os.path.join(label_output_dir, label_file))


# 데이터셋 분리 실행
split_dataset('../dataset/train', './labels/train', './yolo_dataset/train', './yolo_dataset/val')
