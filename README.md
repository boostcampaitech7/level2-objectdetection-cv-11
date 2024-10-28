
# 재활용 품목 분류를 위한 Object Detection

##  Overview
<!-- 재활용 품목 분류를 위한 Object Detection -->
본 프로젝트는 현재의 쓰레기 처리 문제와 환경 위기에 대응하기 위해 2차원 이미지 기반의 쓰레기 분류 모델을 개발하는 것을 목표로 한다. 데이터의 품질과 클래스 간 불균형 문제를 탐색적 데이터 분석(EDA)을 통해 파악하였으며, 이러한 문제를 해결하고자 전략적인 데이터 정제, 학습 데이터 재분류, 데이터 증강, 초해상도 처리, 디블러링 및 클래스 재분류 등을 포함하는 여러 접근 방식을 제안하였다. 다양한 모델 학습 및 앙상블 기법을 적용하여 각 전략을 탐색적으로 실험하고, 결과를 비교 분석하였다. 최종적으로 가장 유의미한 성과를 보인 전략을 채택하여 모델 성능을 최적화하고, 이에 따른 최종(Private) 6위의 성과를 달성하였다.

##  Rank

|Score|Our Score|
|:-------:|:--------------------------------------------------------------:|
|Public| 5위 <img width="800" src="https://github.com/user-attachments/assets/b7f049e5-e06a-4043-a19d-bfa5fc6d0518">
|Private| 6위 <img width="800" src="https://github.com/user-attachments/assets/27aeb60c-8168-4504-adfd-84385a6d44a3">

  

## Dataset


<center>
<div align="center">
  <img src="https://github.com/user-attachments/assets/f0eae989-bb4c-4b5a-ac78-d2081f814259" width="400" height="">
  <br>
  <sup>Example image data with 2D Bounding Boxes</sup>
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

> ### Class Imbalance

<div align="center">
  <table>
    <tr>
      <td>
        <figure>
          <img src="https://github.com/user-attachments/assets/9e9448cf-8475-4cc1-80c5-75ed45fc5db8" width="600">
          <figcaption>A distribution graph showing the number of annotations per class</figcaption>
        </figure>
      </td>
      <td>
        <figure>
        <img src="https://github.com/user-attachments/assets/8d20b22d-6f3a-494c-abc8-af0634079378" width="600">
        <figcaption>Class distribution graph for each fold split with k=5</figcaption>
        </figure>
      </td>
    </tr>
  </table>
</div>


- 분포를 확인한 결과, 클래스 별 불균형이 심하여 KFold 대신 Train/Validation에 동일한 분포를 적용하기 위해 Stratified K-Fold 적용하였다.
- Stratified group k-fold (k=5)를 적용하여 5쌍의 train, validation set을 구성하여 학습에 적용하였다.



> ### Object Position

<div align="center">
  <table>
    <tr>
      <td>
        <figure>
          <img src="https://github.com/user-attachments/assets/8b951206-eab6-4509-b766-e066bdf5a9a5" width="1300">
          <figcaption>Heatmaps Showing Bounding Box Placement</figcaption>
        </figure>
      </td>
      <td>
        <figure>
        <img src="https://github.com/user-attachments/assets/751f771a-223e-4aae-b6aa-37fd19530d01" width="2000">
        <figcaption>A heatmap based on the center points of bounding boxes for each class</figcaption>
        </figure>
      </td>
    </tr>
  </table>
</div>


- 모든 클래스에서 이미지 내 BBOX들의 위치 분포를 분석한 결과, 대부분의 BBOX(객체)가 이미지 중심부에 주로 위치하고 있는 경향이 나타났다.


> ### Aspect Ratio for bounding boxes by class


<center>
<div align="center">
  <img src="https://github.com/user-attachments/assets/6e1f1924-c2bc-43aa-ada5-b24794f02396" width="1000" height="">
</div>
</center>

- 모든 클래스에서 평균 Aspect Ratio가 1보다 크다.
- Aspect ratio > 1: 가로가 세로보다 긴 BBOX를 의미한다. 즉, 가로로 길쭉한 모양이다.

  

> ### DDQ initial Fine Tuning 모델을 사용해 학습 후 결과를 EDA 진행


<div align="center">
  <table>
    <tr>
      <td>
        <figure>
          <img src="https://github.com/user-attachments/assets/42b438da-3464-4afd-98b2-256fe93b3686" width="600">
          <figcaption>Class별 Classification 지표 ; Confusion Matrix</figcaption>
        </figure>
      </td>
      <td>
        <figure>
        <img src="https://github.com/user-attachments/assets/97ec5cd4-6736-4d57-bd07-908b1ddfd0b5" width="600">
        <figcaption>Class별 Classification 지표</figcaption>
        </figure>
      </td>
    </tr>
  </table>
</div>

- Plastic, general trash의 recall을 늘리기 위한 전략이 필요하다.
- (Plastic과 Glass) (General trash와 Plastic Bag)간의 혼동 비율을 감소하는 전략이 필요하다.
- General Trash와 Paper를 못 잡는 경우를 줄이는 전략이 필요하다.
  

<center>
<div align="center">
  <img src="https://github.com/user-attachments/assets/e56f8e18-4567-41e4-8489-f429fae4e595" width="500" height="">
</div>
</center>

- 대부분 작은 object들을 잘 detect 하지 못했다. (EX : 쓰레기 봉투 안에 있는 쓰레기들)





##  Pipeline
<center>
<div align="center">
  <img src="https://github.com/user-attachments/assets/37cd6586-a033-426f-8b1b-8c504e229c8e" width="700" height="" >
  <sup>Pipeline of Applied Methods 
</sup>
</div>
</center>


## Methods

> ### Super Resolution

- 탐색적 데이터 분석(EDA) 결과, 데이터셋 내에 크기가 작고 저화질의 이미지들이 다수 포함되어 있음을 확인하였다.
- 이를 개선하기 위해, 단일 이미지 초해상도 기법으로 Enhanced Deep Residual Networks for Single Image Super-Resolution 논문에서 제안된 방법을 활용하여 이미지를 2배 해상도로 변환하였다.
- 변환된 이미지는 Center-crop 및 Quarter-crop 방식으로 추가 전처리하여, 기존 학습 데이터와 결합하여 모델 학습에 활용하였다.
- 실험 결과, Center-crop 방식(이미지 1개 추가)에 비해 Quarter-crop 방식을 통해 4배의 학습 데이터를 확보한 경우, 성능이 더 크게 향상됨을 확인할 수 있었다.

<center>
<div align="center">
  <img src="https://github.com/user-attachments/assets/dc254fe5-c61a-49c8-a9c8-4eb10154ad17" width="400" height="">
  <sup>Center-Crop</sup>
</div>
</center>


<center>
<div align="center">
    <img src="https://github.com/user-attachments/assets/24a1d669-1e79-456b-baf1-e8c28e00a53a" width="400" height="">
    <sup>Quarter-Crop</sup>
</div>
</center>


<!-- <center>  -->

| Dataset            | Model | Backbone | Epoch | mAP_50(Val) |
|:-------:|:----------:|:-----------------------:|:-------:|:---------:|
| Original           | DINO  | Swin-L   | 12 |  0.7107 |
| Original + Center-Crop SR | DINO  | Swin-L   | 12 | 0.7138  |
| Orginal + Quarter-Crop SR | DINO  | Swin-L   | 12 | 0.7182  |

<!-- </center> -->

> ### Copy Paste with Mosaic
- Copy-Paste는 instance segmetation 분야에서 다양한 model architecture에 robust하며 학습의 효율을 높이고 성능 향상을 이룬 augmentation 기법이다.
- mmdetection에 구현된 copy paste는 다른 image의 object가 위치와 크기가 그대로 해당 image에 포개지는 형태였기에 기존 image의 object가 copy paste된 이미지로 인해 가려지는 현상이 발생했다.
- 이미지를 4분할해 조합하는 Augmentation 기법인 Mosaic에서 영감을 받아 Copy Paste를 Mosaic 방식으로 변형해 copy paste object가 기존의 object를 가리는 현상을 최소화 하고 작은 bbox들이 image의 edge에 분포되도록 유도하였다.
- copy paste는 특히 2stage model에서 좋은 효과가 있었으며 Psuedo Labelling과 함께 활요할 때 더 효과적이었다.

| **Experiment**      | **Model**                 | **mAP_50(Val)** |
|:-----------------------:|:--------------------------:|:-----------:|
| Original    | DDQ          | 0.4939     | 
| Copy Paste  | DDQ          | 0.5486     |

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

> ### Pseudo Labeling
- 본 프로젝트에서는 객체 탐지 성능을 향상시키기 위해 1-stage model인 YOLO11에 Pseudo labeling 기법을 적용하고 그 효과를 분석하였다.
- 더 나은 성능 향상을 위하여 팀 내 가장 예측 성능이 우수한 DINO를 사용하여 Pseudo labeling을 적용하였다.
- Pseudo labeling 된 데이터의 학습 방식에 따른 성능 비교와 Pseudo labeling 시 신뢰도 임계값(confidence threshold)이 모델 성능에 미치는 영향을 분석하였다.
- 실험 결과, 객체 탐지 작업에서 Pseudo labeling을 적용할 때 개별 레이블의 품질보다 학습 데이터의 전체적인 양이 모델 성능 향상에 더 중요한 요인임을 알 수 있었다. 또한, 낮은 신뢰도의 예측 결과도 학습 데이터로 포함시키는 것이 모델의 일반화 성능 향상에 기여할 수 있음을 보여준다.
  
| **Experiment**      | **Model**                 | **mAP_50(Val)** |
|:-----------------------:|:--------------------------:|:-----------:|
| sequential learning      | YOLOv11               | 0.5911     | 
| Learning at once         | YOLOv11               | 0.5663     |

| **Confidence Score**      | **Model**                 | **mAP_50(Val)** |
|:-----------------------:|:--------------------------:|:-----------:|
| 0.25        | YOLOv11               | 0.6320     | 
| 0.5         | YOLOv11               | 0.5911     |
| 0.7         | YOLOv11               | 0.5247     |

> ### TTA
- Test Time Augmentation을 통해 모델의 예측값의 일반화된 성능 향상을 이뤄낼 수 있었다.

| **Experiment**      | **Model**                 | **mAP_50(Val)** |
|:-----------------------:|:--------------------------:|:-----------:|
| No TTA     | DDQ               | 0.6160     | 
| TTA        | DDQ               | 0.6446     |



</br>


> ### Models
- 1-stage 및 2-stage 모델을 포함하여 레거시와 최신 모델을 모두 활용하여 성능을 비교하였다.
- YOLO와 같은 1-stage 모델은 상대적으로 낮은 객체 검출 성능을 보이는 경향이 있었다.
- 최신 연구에서 제안된 DINO, Co-DETR, ATSS 등의 모델을 학습하고 평가하여, 해당 모델들의 성능을 분석하였다.
- 성능 확인 결과 DINO와 DDQ의 성능이 가장 높았기에 두 모델의 성능을 고도화하여 앙상블에 활용하고자 하였다.
```bash
Frameworks : Detectron2 v0.6, Ultralytics v8.1, mmDetection v3.3.0
```





|    **Framework**   |     **Model**    |   **Backbone**   | **Val mAP50** | **Check** |
|:--------------:|:------------:|:------------:|:---------:|:---------:|
| Detectron 2    | Faster RCNN  | R50          |   0.450   |        |
|                | Cascade RCNN |              |   0.452   |        |
| Yolo v8        | Yolo v8m     | CSPDarknet53 |   0.414   |        |
|                | Yolo v8x     |              |   0.474   |        |
| mmDetection v3 | Cascade RCNN | R50          |   0.458   |        |
|                |              | ConvNext-s   |   0.554   |        |
|                |              | Swin-t       |   0.512   |        |
|                | DDQ          | R50          |   0.560   |        |
|                |              | Swin-l       |   0.677   |   ✅     |
|                | DINO         | R101         |   0.580   |        |
|                |              | Swin-l       |   0.719   |   ✅     |
|                | Co-Detr      | Swin-l       |   0.665   |        |

</br>


> ### Ensemble
- 모델별 Confusion Matrix를 분석하여 각 모델의 특성을 파악하고, 최적의 모델 조합을 결정하였다.
- 최종 앙상블 모델 조합은 ATSS, DINO, DDQ를 활용한 조합이었다.
- WBF (Weighted Box Fusion) 기법을 적용했으나 성능 향상에는 큰 효과가 없었다.
- 최종적으로 NMS (Non-Maximum Suppression) 기법을 활용하여 최종 제출 결과물을 생성하였다.

| Models                   | Ensemble mAP_50(Test) |
|:-------------------------------:|:-------------:|
|DDQ + DINO |   0.7410   |
|**DDQ + DINO + ATSS**   |   **0.7423**   |
|DDQ + DINO + ATSS + YOLO11  |   0.7003   |
|DDQ + DINO + ATSS + Cascade Mask RCNN  |   0.7294   |
|DDQ + DINO + Cascade Mask RCNN  |   0.7293   |

| Ensemble Technology             | Ensemble mAP_50(Test) |
|:-------------------------------:|:-------------:|
|**NMS** |   **0.7423**   |
|NMW(1:2:1:1)   |   **0.7373**   |
|NMW(1:0.5:0.5:1)  |   0.7349   |
|WBF  |   0.7072   |

## Conclusion
### LB TImelines
위와 같은 과정을 거치며 mAP50 값이 우상향하는 그래프를 그리며 성능을 향상시킬 수 있었다.
 <center>
<img src="https://github.com/user-attachments/assets/027aed8a-b356-4927-a5c5-fa60aed3d31c" width="700" height="">
<div align="center">
   <sup>LB Timelines Graph. 전체적으로 mAP를 우상향시키는 방향으로 프로젝트가 진행되었다.(Public)
</sup>
</div>
</center>

### Contribution
1. 데이터 증강 및 기법 적용 효과 Augmentation, Copy Paste, Pseudo Labeling, Stratified KFold, Super Resolution 등의 기법을 도입함으로써 모델 성능을 크게 향상
2. 모델 구조 활용 2-Stage 모델(DINO, CO-DETR, DDQ, Cascade Mask R-CNN)은 높은 정확도와 복잡한 장면에서 우수한 성능을 보였으며, 1-Stage 모델(YOLO, UniverseNet, ATSS)은 빠른 학습 속도를 바탕으로 다양한 실험에서 활용
3. 앙상블 기법 성능: DINO, DDQ, ATSS 모델들을 적절히 앙상블 하여 0.7423이라는 자체 최고 성능 달성 및 전체 순위 6위 달성
4. 향후 과제: Co-DETR와 UniverseNet 모델을 앙상블 실험에 포함하지 못한 아쉬움이 있으며, 추가적인 실험과 데이터 분석을 통해 성능을 지속적으로 개선할 예정