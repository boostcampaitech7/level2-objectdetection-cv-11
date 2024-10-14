#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# numpy downgrade가 필요할 수 있음
# !python3 -m pip install numpy==1.20.0


# In[ ]:


# !pip install ensemble_boxes


# In[ ]:


CUDA_VISIBLE_DEVICES = 1


# In[1]:


import pandas as pd
from pycocotools.coco import COCO
import numpy as np
from ensemble_boxes import *


# In[2]:


# ensemble csv files
submission_files = [
    '/home/donghun0671/workplace/lv2/level2-objectdetection-cv-11/mmdetection/work_dirs/recycle_dino-5scale_swin-l_8xb2-6e_coco_copy_0/20241013_205130/submit/submission.csv',
    '/home/donghun0671/workplace/lv2/level2-objectdetection-cv-11/mmdetection/work_dirs/recycle_dino-5scale_swin-l_8xb2-6e_coco_copy_1/20241011_154707/submit/submission.csv',
    '/home/donghun0671/workplace/lv2/level2-objectdetection-cv-11/mmdetection/work_dirs/recycle_dino-5scale_swin-l_8xb2-6e_coco_copy_2/20241013_204857/submit/submission.csv',
    '/home/donghun0671/workplace/lv2/level2-objectdetection-cv-11/mmdetection/work_dirs/recycle_dino-5scale_swin-l_8xb2-6e_coco_copy_3/20241013_204935/submit/submission.csv',
    '/home/donghun0671/workplace/lv2/level2-objectdetection-cv-11/mmdetection/work_dirs/recycle_dino-5scale_swin-l_8xb2-6e_coco_copy_4/20241013_204959/submit/submission.csv'
    ] # submission csv 파일 이름을 넣어주세용

submission_df = [pd.read_csv(file) for file in submission_files]


# In[3]:


image_ids = submission_df[0]['image_id'].tolist()


# In[ ]:


annotation = '/home/donghun0671/workplace/lv2/dataset/test.json' # test.json 경로 설정!
coco = COCO(annotation)


# In[ ]:


prediction_strings = []
file_names = []

for i, image_id in enumerate(image_ids):
    prediction_string = ''
    boxes_list = []
    scores_list = []
    labels_list = []
    image_info = coco.loadImgs(i)[0]
    
    for df in submission_df:
        predict_string = df[df['image_id'] == image_id]['PredictionString'].tolist()[0]
        predict_list = str(predict_string).split()
        
        if len(predict_list)==0 or len(predict_list)==1:
            continue
            
        predict_list = np.reshape(predict_list, (-1, 6))
        box_list = []
        
        for box in predict_list[:, 2:6].tolist():
            box[0] = float(box[0]) / image_info['width']
            box[1] = float(box[1]) / image_info['height']
            box[2] = float(box[2]) / image_info['width']
            box[3] = float(box[3]) / image_info['height']
            box_list.append(box)
            
        boxes_list.append(box_list)
        scores_list.append(list(map(float, predict_list[:, 1].tolist())))
        labels_list.append(list(map(int, predict_list[:, 0].tolist())))
    
    if len(boxes_list):
        # boxes, scores, labels = nms(boxes_list, scores_list, labels_list,iou_thr=iou_thr)
        # boxes, scores, labels = soft_nms(box_list, scores_list, labels_list, iou_thr=iou_thr)
        # boxes, scores, labels = non_maximum_weighted(boxes_list, scores_list, labels_list,iou_thr=iou_thr)
        # boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list,iou_thr=0.5,conf_type='box_and_model_avg')
        boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list,iou_thr=0.55)

        for box, score, label in zip(boxes, scores, labels):
            prediction_string += str(int(label)) + ' ' + str(score) + ' ' + str(box[0] * image_info['width']) + ' ' + str(box[1] * image_info['height']) + ' ' + str(box[2] * image_info['width']) + ' ' + str(box[3] * image_info['height']) + ' '
    
    prediction_strings.append(prediction_string)
    file_names.append(image_id)


# In[ ]:


submission = pd.DataFrame()
submission['PredictionString'] = prediction_strings
submission['image_id'] = file_names
try:
    submission.to_csv('/home/donghun0671/workplace/lv2/level2-objectdetection-cv-11/tools/16tools/dino_submission_ensemble_wbf.csv', index=False)
    print("파일 저장 성공!")
except Exception as e:
    print(f"파일 저장 중 오류 발생: {e}")


submission.head()

