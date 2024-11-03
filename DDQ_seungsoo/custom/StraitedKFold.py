#%%
from sklearn.model_selection import StratifiedGroupKFold
import json
import numpy as np
# load json
annotation = '../dataset/train.json'

with open(annotation) as f:
    data = json.load(f)
    
var = [(ann['image_id'], ann['category_id']) for ann in data['annotations']]

X = np.ones((len(data['annotations']),1))
y = np.array([v[1] for v in var])
groups = np.array([v[0] for v in var])

n_splits = 5
cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=411)

trains=[]
valds=[]

for train_idx, val_idx in cv.split(X, y, groups):
    trains.append(groups[train_idx])
    print("TRAIN:", groups[train_idx])
    print("      ", y[train_idx])
    valds.append(groups[val_idx])
    print(" TEST:", groups[val_idx])
    print("      ", y[val_idx])

# %%
def make_json(type = 'annot', fold = 0):
    input_json_path = '../dataset/train.json'

    # COCO 형식의 test.json 파일 읽기
    with open(input_json_path, 'r') as f:
        coco_data = json.load(f)

    output_json_path = '../dataset/evaluation_'+type+'.json'

    # annotations의 fold만 선택
    coco_data['images'] = np.array(coco_data['images'])[trains[fold]].tolist()

    # 관련 이미지 및 카테고리만 남기기 위한 필터링
    annotation_image_ids = {ann['id'] for ann in coco_data['images']}

    # 해당 annotation과 관련된 image들만 남김
    if type!='annot':
        coco_data['annotations']=[]
    else:
        coco_data['annotations'] = [img for img in coco_data['annotations'] if img['image_id'] in annotation_image_ids]

    # 50개 annotation만 포함된 새로운 json 파일 저장
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=4)

    print(f"Saved 50 annotations to {output_json_path}")

if __name__ == '__main__':  
    make_json()
    make_json(type = 'nonannot')