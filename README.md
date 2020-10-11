# Face-Member
얼굴인식 기반 회원 기능

1. signup.py : 탐지된 얼굴 100장 찍은 후 faces/사용자명/ 디렉터리에 저장
2. train.py : faces/사용자명/ 디렉터리의 사진을 학습 후 models/사용자명Data.yml 로 저장
3. run : models/ 디렉터리의 모델 파일들을 불러와 사용자 판별 후 출력
