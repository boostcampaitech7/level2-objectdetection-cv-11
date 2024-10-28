import json
import os
import random
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold



def split_dataset(input_json, output_dir, random_seed):
    random.seed(random_seed)

    with open(input_json) as json_reader:
        dataset = json.load(json_reader)

    images = dataset['images']
    annotations = dataset['annotations']
    categories = dataset['categories']

    k_var = [(ann['image_id'], ann['category_id']) for ann in annotations]
    X = np.ones((len(annotations),1))
    y = np.array([v[1] for v in k_var])
    groups = np.array([v[0] for v in k_var])
    
    #file_name에 prefix 디렉토리까지 포함 (CocoDataset 클래스를 사용하는 경우)
    #for image in images:
    #    image['file_name'] = '{}/{}'.format(image['file_name'][0], image['file_name'])
        
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=411)
    
    for k, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        train_ids = groups[train_idx]
        val_ids = groups[val_idx]
        
        image_ids_val, image_ids_train = set(val_ids), set(train_ids)

        train_images = [x for x in images if x.get('id') in image_ids_train]
        val_images = [x for x in images if x.get('id') in image_ids_val]
        train_annotations = [x for x in annotations if x.get('image_id') in image_ids_train]
        val_annotations = [x for x in annotations if x.get('image_id') in image_ids_val]

        train_data = {
            'images': train_images,
            'annotations': train_annotations,
            'categories': categories,
        }

        val_data = {
            'images': val_images,
            'annotations': val_annotations,
            'categories': categories,
        }

        output_seed_dir = os.path.join(output_dir, f'seed{random_seed}')
        os.makedirs(output_seed_dir, exist_ok=True)
        output_train_json = os.path.join(output_seed_dir, f'train_{k}.json')
        output_val_json = os.path.join(output_seed_dir, f'val_{k}.json')

        print(f'write {output_train_json}')
        with open(output_train_json, 'w') as train_writer:
            json.dump(train_data, train_writer)

        print(f'write {output_val_json}')
        with open(output_val_json, 'w') as val_writer:
            json.dump(val_data, val_writer)

        #print(f'write {output_train_csv}, {output_val_csv}')
        #with open(input_csv, 'r') as csv_reader, \
        #        open(output_train_csv, 'w') as train_writer, \
        #        open(output_val_csv, 'w') as val_writer:
        #    train_writer.write('ImageId,EncodedPixels,Height,Width,CategoryId\n')
        #    val_writer.write('ImageId,EncodedPixels,Height,Width,CategoryId\n')
        #    for line in csv_reader:
        #        if line.startswith('ImageId'): continue
        #        image_id, encoded_pixels, height, width, category_id = line.strip().split(',')
        #        image_id = int(image_id)
        #        if image_id in image_ids_train:
        #            train_writer.write(line)
        #        elif image_id in image_ids_val:
        #            val_writer.write(line)
        #        else:
        #            raise ValueError(f'unknown image_id: {image_id}')
        
split_dataset("../../../dataset/train.json", "../../../dataset", 24 )
