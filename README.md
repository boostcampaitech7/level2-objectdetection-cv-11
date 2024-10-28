
# 재활용 품목 분류를 위한 Object Detection

##  Overview
<!-- 재활용 품목 분류를 위한 Object Detection -->
본 프로젝트는 현재의 쓰레기 처리 문제와 환경 위기에 대응하기 위해 2차원 이미지 기반의 쓰레기 분류 모델을 개발하는 것을 목표로 한다. 데이터의 품질과 클래스 간 불균형 문제를 탐색적 데이터 분석(EDA)을 통해 파악하였으며, 이러한 문제를 해결하고자 전략적인 데이터 정제, 학습 데이터 재분류, 데이터 증강, 초해상도 처리, 디블러링 및 클래스 재분류 등을 포함하는 여러 접근 방식을 제안하였다. 다양한 모델 학습 및 앙상블 기법을 적용하여 각 전략을 탐색적으로 실험하고, 결과를 비교 분석하였다. 최종적으로 가장 유의미한 성과를 보인 전략을 채택하여 모델 성능을 최적화하고, 이에 따른 n위의 성과를 달성하였다.

##  Rank

<center>
<img src="" width="700" height="">
<div align="center">
  <sup>Test dataset(Public)
  <img width="721" alt="중간순위" src="https://github.com/user-attachments/assets/b7f049e5-e06a-4043-a19d-bfa5fc6d0518">

</sup>
</div>
</center>

<center>
<img src="" width="700" height="">
<div align="center">
  <sup>Test dataset(Private)
  ![최종 순위](https://github.com/user-attachments/assets/dc2da23a-a6b8-416b-85a8-35dd2934d2ac)  
</sup>
</div>
</center>


## Dataset


<center>
<img src="https://github.com/Eddie-JUB/Portfolio/assets/71426994/474d0b88-6d5b-43b3-84df-b18b858a17ad" width="700" height="">
<div align="center">
  <sup>Example image data w/ 2D Bounding Boxes, annotation data
</sup>
</div>
</center>

![대회 개요](https://github.com/user-attachments/assets/459d801e-cfa0-438a-8721-fd9b452f4d6b)


- **Images & Size :**   9754장(train 4883, test 4871) & (1024, 1024)
- **classes :** General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
<!-- - **Annotations :** Image size, class,  -->

<!-- <br/> -->

## Members 
<table>
    <tr height="160px">
        <td align="center" width="150px">
            <a href="https://github.com/naringles"><img height="110px"  src="https://avatars.githubusercontent.com/u/61579399?v=4"></a>
            <br/>
            <a href="https://github.com/naringles"><strong>임동훈</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/hanseungsoo13"><img height="110px"  src="https://avatars.githubusercontent.com/u/75753717?v=4"/></a>
            <br/>
            <a href="https://github.com/hanseungsoo13"><strong>한승수</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/Jeong-AYeong"><img height="110px"  src="https://avatars.githubusercontent.com/u/87751593?v=4"/></a>
            <br/>
            <a href="https://github.com/Jeong-AYeong"><strong>정아영</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/Ai-BT"><img height="110px" src="https://avatars.githubusercontent.com/u/97381138?v=4"/></a>
            <br />
            <a href="https://github.com/Ai-BT"><strong>김대환</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/cherry-space"><img height="110px" src="https://avatars.githubusercontent.com/u/177336350?v=4"/></a>
            <br />
            <a href="https://github.com/cherry-space"><strong>김채리</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
              <a href="https://github.com/SkyBlue-boy"><img height="110px"  src="https://avatars.githubusercontent.com/u/63849988?v=4"/></a>
              <br />
              <a href="https://github.com/SkyBlue-boy"><strong>박윤준</strong></a>
              <br />
          </td>
    </tr>
</table>  
      
                

##  Roles

|Name|Roles|
|:-------:|:--------------------------------------------------------------:|
|Common||
|임동훈| DINO, KFold, Ensemble, Augmentation, Super Resolution
|한승수| DDQ, Ensemble, Custom Copy Paste, EDA
|정아영| YOLOv8, YOLOv11, Pseudo Labeling
|김대환| Moderator, YOLOv7, YOLOv8, YOLO MultiScaling
|김채리| ATSS, ConvNext, LR Scheduler
|박윤준| EDA, UniverseNet


</br>

##  Enviroments

- Language: Python 3.8
- Hardwares: Intel(R) Xeon(R) Gold 5120, Tesla V100-SXM2 32GB × 4
- Framework: Pytorch, Detectron2 v0.6, Ultralytics v8.1, MMDetection v3.3.0
- Cowork Tools: Github, Weight and Bias, Notion, Zoom

</br>

#  Project
##  EDA

<!-- <center>
<img src="https://github.com/Eddie-JUB/Portfolio/assets/71426994/4dd0e517-83dc-4d7e-9c3f-c6d7db10ed7c" width="700" height="">
</center> -->


> ### Class Imbalance, Object Size

<center>
<img src="" width="900" height="">
<div align="center">
  <sup>Distribution of Bbox area as % of Image area by class
</sup>
</div>
</center>


데이터셋에서 Paper, Plastic bag, General trash 클래스가 높은 비중을 차지하고 있어, 전반적으로 데이터의 분포가 불균형함을 확인하였다.
클래스별 객체의 크기 분포를 분석한 결과, 작은 객체의 빈도는 높았으며, 객체 크기가 커질수록 해당 객체 수가 감소하는 경향을 보였다.

</br>

> ### Object Position

<center>
<img src="">
<div align="center">
  <sup>Object Bounding Box distribution of each class
</sup>
</div>
</center>


- 이미지 내 객체들의 위치 분포를 분석한 결과, 대부분의 객체가 이미지 중심부에 주로 위치하고 있는 경향이 나타났다.
</br>

##  Pipeline
<center>
<img src="" >
<div align="center">
  <sup>Pipeline of Applied Methods 
</sup>
</div>
</center>


## Methods

> ### Super Resolution

- 탐색적 데이터 분석(EDA) 결과, 데이터셋 내에 크기가 작고 저화질의 이미지들이 다수 포함되어 있음을 확인하였다.
- 이를 개선하기 위해, 단일 이미지 초해상도 기법으로 Enhanced Deep Residual Networks for Single Image Super-Resolution 논문에서 제안된 방법을 활용하여 이미지를 2배 해상도로 변환하였다.
- 변환된 이미지는 Center-crop 및 Multi-crop 방식으로 추가 전처리하여, 기존 학습 데이터와 결합하여 모델 학습에 활용하였다.
- 실험 결과, Center-crop 방식(이미지 1개 추가)에 비해 Multi-crop 방식을 통해 4배의 학습 데이터를 확보한 경우, 성능이 더 크게 향상됨을 확인할 수 있었다.

<center>
<img src="" width="400" height="">
<div align="center">
  <sup>Center-Crop</sup>
</div>
</center>


<center>
<img src="" width="400" height="">
<div align="center">
    <sup>Multi-Crop</sup>
</div>
</center>


<!-- <center>  -->

| Dataset            | Model | Backbone | Epoch | mAP_50(Val) | mAP_50(Test)
|:-------:|:----------:|:-----------------------:|:-------:|:---------:|:---------:|
| Original           | DINO  | Swin-l   | |   | |
| Original+SR(Center-Crop)      | DINO  | Swin-l   | |   | |
| Original+SR(Multi-Crop) | DINO  | Swin-l   | |   | |

<!-- </center> -->
    

</br>

> ### Augmentation
- 객체의 크기와 위치를 고려하여, 데이터 증강 기법으로 RandomResize, RandomCrop, RandAugment를 적용함으로써 성능 개선 효과를 기대하였다.
- 모델의 일반화 성능을 향상시키기 위해 다양한 증강 기법을 적용하였으며, 여러 평가 지표를 바탕으로 최적의 증강 기법을 선정하였다.
- 기하학적 변환 기법을 적용했을 때 IoU 임계값에 따라 mAP가 크게 변동하는 경향이 관찰되었다.
- 색상 변환에 RandAugment 기법을 적용한 결과, 모델의 강건성과 성능이 유의미하게 향상되는 것을 확인할 수 있었다.

| **Augmentation**      | **Info**                 | **mAP_50(Val)** |
|:-----------------------:|:--------------------------:|:-----------:|
| None                  | -                        | 0.553     | 
| RandomCrop            | RandomCrop               | 0.564     |
| RandomCenterCropPad   | CenterCrop + pad         | 0.567     |
| RandomAffine          | Geometric transformation | 0.560     |
| PhotoMetricDistortion | Color Jitter             | 0.563     |
| RandAugment           | Color transformation     | 0.570     |


</br>


> ### Models
- 1-stage 및 2-stage 모델을 포함하여 레거시와 최신 모델을 모두 활용하여 성능을 비교하였다.
- YOLO와 같은 1-stage 모델은 상대적으로 낮은 객체 검출 성능을 보이는 경향이 있었다.
- 최신 연구에서 제안된 DINO, Co-DETR 등의 모델을 학습하고 평가하여, 해당 모델들의 성능을 분석하였다.
```bash
Frameworks : Detectron2 v0.6, Ultralytics v8.1, mmDetection v3.3.0
```
내용추가!
<!-- <center>
<img src="https://github.com/FinalCold/Programmers/assets/67350632/6de0cd46-8ee8-4f85-a9cf-1215d2d453fd" width="700" height="">
<div align="center"> -->
<!--   <sup>Test dataset(Public) -->
<!-- </sup>
</div>
</center> -->





<!-- |    Framework   |     Model    |   Backbone   | Val mAP50 |
|:--------------:|:------------:|:------------:|:---------:|
| Detectron 2    | Faster RCNN  | R50          |   0.450   |
|                | Cascade RCNN |              |   0.452   |
| Yolo v8        | Yolo v8m     | CSPDarknet53 |   0.414   |
|                | Yolo v8x     |              |   0.474   |
| mmDetection v3 | Cascade RCNN | R50          |   0.458   |
|                |              | ConvNext-s   |   0.554   |
|                |              | Swin-t       |   0.512   |
|                | DDQ          | R50          |   0.560   |
|                |              | Swin-l       |   0.677   |
|                | DINO         | R101         |   0.580   |
|                |              | Swin-l       |   0.719   |
|                | Co-Detr      | Swin-l       |   0.717   |
 -->

</br>


> ### Ensemble
- 모델별 Confusion Matrix를 분석하여 각 모델의 특성을 파악하고, 최적의 모델 조합을 결정하였다.
- WBF (Weighted Box Fusion) 기법을 적용했으나 성능 향상에는 큰 효과가 없었다.
- 최종적으로 NMS (Non-Maximum Suppression) 기법을 활용하여 최종 제출 결과물을 생성하였다.

| Models                   | Average mAP_50(Val) | Ensemble mAP_50(Test) |
|:-------------------------------:|:-------------:|:-----------------:|
|내용추가 |      |       |
|내용추가   |      |       |
