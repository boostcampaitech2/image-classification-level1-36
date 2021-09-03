# 마스크 착용 상태 분류

### Task Description: 주어진 이미지 속 사람의 마스크 착용 여부, 성별, 나이대를 예측해 총 18개의 클래스로 분류
<img src=https://cdn.pixabay.com/photo/2020/07/14/19/12/wearing-a-mandatory-mask-5405387_1280.png  width="400" height="370">

---

### 실행 방법 Getting started:

1. Install RetinaFace for face crop

  ```shell
  pip install -r requirements.txt
  pip install git+https://github.com/elliottzheng/face-detection.git@master

  ```


2. Crop Faces out of Train & Eval Data
  ```shell
  cd utils
  python3 crop.py
  ```
  
3. 
