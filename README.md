# ouput.csv로 Pseudo labeling 하는 법  
### 1. make_pseudo_label.ipynb를 통해 원하는 threshold에 해당하는 bbox정보만 추출
### 2. make_pseudo_json_final.ipynb실행
- 1에서 추출한 bbox를 json형태로 변환하고 train이미지와 (bbox가 존재하는)test이미지를 하나의 새로운 폴더에 저장
  ![image](https://github.com/user-attachments/assets/656a919a-fd65-40d1-8e5c-154340d3919d)

### 3. coco2yolo2pseudo.py 실행
- json파일의 bbox정보를 yolo label 형태인 txt파일로 저장

  ![image](https://github.com/user-attachments/assets/ad102347-fef6-47c3-a6f7-be8684b0f776)
### 4. 바뀐 데이터에 맞게 yaml파일 수정
- 예시는 config폴더에서 확인가능
- 경로 설정 주의
### 5. yolo학습 실행
